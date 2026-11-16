"""Local HTTP service for persistent community review of uncertain sightings."""

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from getpass import getpass
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs
from uuid import UUID, uuid4
from wsgiref.simple_server import make_server
from zipfile import ZipFile

from sqlalchemy import (
    Column, ForeignKey, Index, Integer, LargeBinary, MetaData, Table, Text,
    and_, create_engine, event, func, select,
)
from sqlalchemy.engine import URL

from serpentguard.community_voting import ROLE_WEIGHT, RULE_VERSION, SightingBallot
from serpentguard.voting_evidence import (
    CAPTURE_FIELDS, MAX_EVIDENCE_BYTES, PREDICTION_FIELDS, is_confidence, sanitized_jpeg, validate_capture,
)


SCHEMA = MetaData()
USERS = Table(
    "users", SCHEMA,
    Column("id", Text, primary_key=True),
    Column("role", Text, nullable=False),
    Column("token_digest", Text, nullable=False, unique=True),
)
SIGHTINGS = Table(
    "sightings", SCHEMA,
    Column("sequence", Integer, primary_key=True),
    Column("id", Text, nullable=False, unique=True),
    Column("reporter_id", Text, ForeignKey("users.id"), nullable=False),
    Column("payload", Text, nullable=False),
    Column("consensus", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="pending"),
)
VOTES = Table(
    "votes", SCHEMA,
    Column("sequence", Integer, primary_key=True),
    Column("id", Text, nullable=False, unique=True),
    Column("sighting_id", Text, ForeignKey("sightings.id"), nullable=False),
    Column("reviewer_id", Text, ForeignKey("users.id"), nullable=False),
    Column("rank", Text, nullable=False),
    Column("label", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("previous_vote_id", Text, ForeignKey("votes.id")),
)
EVIDENCE = Table(
    "sighting_evidence", SCHEMA,
    Column("sighting_id", Text, ForeignKey("sightings.id"), primary_key=True),
    Column("slot", Integer, primary_key=True),
    Column("jpeg", LargeBinary, nullable=False),
)
CURATORS = Table(
    "training_curators", SCHEMA,
    Column("user_id", Text, ForeignKey("users.id"), primary_key=True),
)
APPROVALS = Table(
    "training_approvals", SCHEMA,
    Column("sighting_id", Text, ForeignKey("sightings.id"), primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("curator_id", Text, ForeignKey("training_curators.user_id"), nullable=False),
    Column("approved_at", Text, nullable=False),
    Column("consensus", Text, nullable=False),
)
Index("review_feed", SIGHTINGS.c.status, SIGHTINGS.c.sequence)
Index("current_votes", VOTES.c.sighting_id, VOTES.c.reviewer_id, VOTES.c.sequence)

INITIAL_CONSENSUS = {
    "status": "pending", "rank": None, "label": None,
    "posterior_confidence": None, "reviewer_count": 0,
    "support_weight": {"genus": {}, "species": {}}, "revision": 0,
}
CONSENSUS_STATUSES = ("species_consensus", "genus_consensus")
CURRENT_APPROVAL = and_(
    APPROVALS.c.sighting_id == SIGHTINGS.c.id,
    APPROVALS.c.revision == func.json_extract(SIGHTINGS.c.consensus, "$.revision"),
)
TRAINING_ELIGIBLE = and_(
    SIGHTINGS.c.status.in_(CONSENSUS_STATUSES),
    func.json_type(SIGHTINGS.c.payload, "$.training_consent") == "true",
)


class RequestError(Exception):
    def __init__(self, status, message):
        self.status = HTTPStatus(status)
        super().__init__(message)


def create_database_engine(path):
    engine = create_engine(URL.create("sqlite", database=str(path)))

    @event.listens_for(engine, "engine_connect")
    def enable_foreign_keys(connection):
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        connection.commit()

    return engine


def provision_user(path, role, *, training_curator=False):
    if role not in ROLE_WEIGHT:
        raise ValueError("Unknown reviewer role")
    user_id, token = str(uuid4()), secrets.token_urlsafe(32)
    engine = create_database_engine(path)
    try:
        SCHEMA.create_all(engine)
        with engine.begin() as db:
            db.execute(USERS.insert().values(id=user_id, role=role, token_digest=token_digest(token)))
            if training_curator:
                db.execute(CURATORS.insert().values(user_id=user_id))
    finally:
        engine.dispose()
    return {"user_id": user_id, "role": role, "token": token}


def token_digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def authenticate(db, environ):
    scheme, _, token = environ.get("HTTP_AUTHORIZATION", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise RequestError(401, "A valid bearer token is required")
    user = db.execute(
        select(USERS.c.id, USERS.c.role).where(USERS.c.token_digest == token_digest(token))
    ).mappings().one_or_none()
    if user is None:
        raise RequestError(401, "A valid bearer token is required")
    return user


def require_curator(db, user):
    if db.scalar(select(CURATORS.c.user_id).where(CURATORS.c.user_id == user["id"])) is None:
        raise RequestError(403, "Training curator access is required")


def parse_uuid(value, field):
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError):
        raise RequestError(400, f"{field} must be a UUID") from None


def read_body(environ, content_type, limit):
    if environ.get("CONTENT_TYPE", "").split(";", 1)[0].strip().lower() != content_type:
        raise RequestError(415, f"Send {content_type}")
    try:
        size = int(environ.get("CONTENT_LENGTH", ""))
    except ValueError:
        raise RequestError(400, "A valid Content-Length is required") from None
    if not 1 <= size <= limit:
        raise RequestError(413, f"Request body must be between 1 and {limit} bytes")
    body = environ["wsgi.input"].read(size)
    if len(body) != size:
        raise RequestError(400, "Request body is shorter than Content-Length")
    return body


def read_json(environ):
    body = read_body(environ, "application/json", 16_384)
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise RequestError(400, "Invalid JSON") from None
    if not isinstance(data, dict):
        raise RequestError(400, "Expected a JSON object")
    return data


def require_fields(data, required, optional=()):
    fields = set(data)
    if not set(required) <= fields or fields - set(required) - set(optional):
        raise RequestError(400, "Missing or unexpected request fields")


def create_sighting(db, user, data):
    require_fields(
        data, ("id", "trigger", "captured_at"),
        ("model_version", "model_confidence", *CAPTURE_FIELDS),
    )
    sighting_id = parse_uuid(data["id"], "id")
    if data["trigger"] not in ("unknown", "low_confidence"):
        raise RequestError(400, "Only uncertain sightings enter this feed")
    try:
        timestamp = datetime.fromisoformat(data["captured_at"])
        if timestamp.utcoffset() is None:
            raise ValueError
        captured_at = timestamp.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        raise RequestError(400, "captured_at must be an ISO 8601 timestamp with timezone") from None
    if "model_version" in data and (
        not isinstance(data["model_version"], str) or not 1 <= len(data["model_version"]) <= 128
    ):
        raise RequestError(400, "model_version must be a nonempty string of at most 128 characters")
    confidence = data.get("model_confidence")
    if confidence is not None and not is_confidence(confidence):
        raise RequestError(400, "model_confidence must be a number between 0 and 1")
    has_evidence = any(field in data for field in CAPTURE_FIELDS)
    if has_evidence:
        try:
            validate_capture(data)
        except ValueError as exc:
            raise RequestError(400, str(exc)) from None
    data = {**data, "id": sighting_id, "captured_at": captured_at}
    payload = json.dumps(data, sort_keys=True)
    existing = db.execute(
        select(SIGHTINGS.c.reporter_id, SIGHTINGS.c.payload).where(SIGHTINGS.c.id == sighting_id)
    ).mappings().one_or_none()
    if existing is not None:
        if (existing["reporter_id"], existing["payload"]) != (user["id"], payload):
            raise RequestError(409, "Sighting ID already exists with different content")
        return 200, data
    db.execute(SIGHTINGS.insert().values(
        id=sighting_id, reporter_id=user["id"], payload=payload,
        consensus=json.dumps(INITIAL_CONSENSUS), status="awaiting_evidence" if has_evidence else "pending",
    ))
    return 201, data


def get_sighting(db, sighting_id):
    row = db.execute(
        select(SIGHTINGS.c.reporter_id, SIGHTINGS.c.consensus, SIGHTINGS.c.payload, SIGHTINGS.c.status)
        .where(SIGHTINGS.c.id == sighting_id)
    ).mappings().one_or_none()
    if row is None:
        raise RequestError(404, "Sighting not found")
    return row


def evidence_path(sighting_id, slot):
    return f"/sightings/{sighting_id}/evidence/{slot}"


def put_evidence(db, user, sighting_id, slot, raw):
    sighting = get_sighting(db, sighting_id)
    if sighting["reporter_id"] != user["id"]:
        raise RequestError(403, "Only the reporter can upload evidence")
    manifest = json.loads(sighting["payload"]).get("evidence", [])
    if slot >= len(manifest):
        raise RequestError(404, "Evidence slot not found")
    if hashlib.sha256(raw).hexdigest() != manifest[slot]["sha256"]:
        raise RequestError(409, "Upload does not match the sighting's evidence SHA-256")
    filters = (EVIDENCE.c.sighting_id == sighting_id, EVIDENCE.c.slot == slot)
    existing = db.scalar(select(EVIDENCE.c.slot).where(*filters))
    path = evidence_path(sighting_id, slot)
    if existing is not None:
        return 200, {"url": path, "ready": sighting["status"] != "awaiting_evidence"}
    try:
        jpeg = sanitized_jpeg(raw)
    except ValueError as exc:
        raise RequestError(422, str(exc)) from None
    db.execute(EVIDENCE.insert().values(sighting_id=sighting_id, slot=slot, jpeg=jpeg))
    uploaded = db.scalar(select(func.count()).select_from(EVIDENCE).where(
        EVIDENCE.c.sighting_id == sighting_id
    ))
    ready = uploaded == len(manifest)
    if ready:
        db.execute(SIGHTINGS.update().where(SIGHTINGS.c.id == sighting_id).values(status="pending"))
    return 201, {"url": path, "ready": ready}


def get_evidence(db, user, sighting_id, slot):
    sighting = get_sighting(db, sighting_id)
    if sighting["status"] == "awaiting_evidence" and sighting["reporter_id"] != user["id"]:
        raise RequestError(404, "Evidence not found")
    jpeg = db.scalar(select(EVIDENCE.c.jpeg).where(
        EVIDENCE.c.sighting_id == sighting_id, EVIDENCE.c.slot == slot,
    ))
    if jpeg is None:
        raise RequestError(404, "Evidence not found")
    return 200, jpeg


def feed_bounds(environ):
    query = parse_qs(environ.get("QUERY_STRING", ""))
    try:
        cursor = int(query.get("cursor", ["0"])[0])
        limit = int(query.get("limit", ["20"])[0])
        if not 0 <= cursor <= 2**63 - 1 or not 1 <= limit <= 100:
            raise ValueError
    except ValueError:
        raise RequestError(400, "Use a nonnegative cursor and a limit of 1–100") from None
    return cursor, limit


def page(items, rows, limit):
    return {"items": items, "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None}


def public_capture(payload):
    item = json.loads(payload)
    item.pop("training_consent", None)
    if "evidence" in item:
        # Keep the model guesses for audit and let reviewers make their own identification.
        for field in PREDICTION_FIELDS:
            item.pop(field, None)
        item["evidence"] = [
            {k: v for k, v in entry.items() if k != "sha256"}
            | {"url": evidence_path(item["id"], slot)}
            for slot, entry in enumerate(item["evidence"])
        ]
    return item


def review_feed(db, user, environ):
    cursor, limit = feed_bounds(environ)
    current_vote = (
        select(func.max(VOTES.c.sequence))
        .where(VOTES.c.sighting_id == SIGHTINGS.c.id, VOTES.c.reviewer_id == user["id"])
        .correlate(SIGHTINGS)
        .scalar_subquery()
    )
    vote = VOTES.alias()
    rows = db.execute(
        select(
            SIGHTINGS.c.sequence, SIGHTINGS.c.payload,
            vote.c.id.label("vote_id"), vote.c.rank.label("vote_rank"), vote.c.label.label("vote_label"),
        )
        .select_from(SIGHTINGS.outerjoin(vote, vote.c.sequence == current_vote))
        .where(SIGHTINGS.c.status == "pending", SIGHTINGS.c.sequence > cursor)
        .order_by(SIGHTINGS.c.sequence)
        .limit(limit + 1)
    ).mappings().all()
    items = []
    for row in rows[:limit]:
        item = public_capture(row["payload"])
        item["your_vote"] = None
        if row["vote_id"] is not None:
            item["your_vote"] = {
                "id": row["vote_id"], "rank": row["vote_rank"], "label": row["vote_label"],
            }
        items.append(item)
    return page(items, rows, limit)


def cast_vote(db, user, sighting_id, data):
    require_fields(data, ("id", "rank", "label"), ("previous_vote_id",))
    vote_id = parse_uuid(data["id"], "id")
    previous_id = data.get("previous_vote_id")
    if previous_id is not None:
        previous_id = parse_uuid(previous_id, "previous_vote_id")
    rank, label = data["rank"], data["label"]
    if not isinstance(rank, str) or not isinstance(label, str):
        raise RequestError(400, "rank and label must be strings")
    sighting = get_sighting(db, sighting_id)
    if sighting["reporter_id"] == user["id"]:
        raise RequestError(403, "Reviewers cannot vote on their own sighting")
    if sighting["status"] == "awaiting_evidence":
        raise RequestError(409, "Wait for the sighting evidence before voting")
    existing = db.execute(
        select(
            VOTES.c.sighting_id, VOTES.c.reviewer_id, VOTES.c.rank, VOTES.c.label, VOTES.c.previous_vote_id,
        ).where(VOTES.c.id == vote_id)
    ).one_or_none()
    if existing is not None:
        if tuple(existing) != (sighting_id, user["id"], rank, label, previous_id):
            raise RequestError(409, "Vote ID already exists with different content")
        return 200, json.loads(sighting["consensus"])

    latest_per_reviewer = (
        select(func.max(VOTES.c.sequence))
        .where(VOTES.c.sighting_id == sighting_id)
        .group_by(VOTES.c.reviewer_id)
    )
    votes = db.execute(
        select(VOTES.c.id, VOTES.c.reviewer_id, VOTES.c.role, VOTES.c.rank, VOTES.c.label)
        .where(VOTES.c.sequence.in_(latest_per_reviewer))
        .order_by(VOTES.c.sequence)
    ).mappings().all()
    active = next((vote["id"] for vote in votes if vote["reviewer_id"] == user["id"]), None)
    if previous_id != active:
        raise RequestError(409, "previous_vote_id must match your current active vote")
    roles = {vote["reviewer_id"]: vote["role"] for vote in votes}
    roles[user["id"]] = user["role"]
    ballot = SightingBallot(roles)
    for vote in votes:
        if vote["reviewer_id"] != user["id"]:
            ballot.cast_vote(vote["reviewer_id"], vote["rank"], vote["label"])
    try:
        ballot.cast_vote(user["id"], rank, label)
    except ValueError as exc:
        raise RequestError(400, str(exc)) from None
    db.execute(VOTES.insert().values(
        id=vote_id, sighting_id=sighting_id, reviewer_id=user["id"], rank=rank,
        label=label, role=user["role"], previous_vote_id=previous_id,
    ))
    result = ballot.result()
    consensus = {field: result[field] for field in INITIAL_CONSENSUS if field != "revision"}
    consensus["revision"] = json.loads(sighting["consensus"])["revision"] + 1
    consensus["rule_version"] = RULE_VERSION
    db.execute(
        SIGHTINGS.update().where(SIGHTINGS.c.id == sighting_id)
        .values(consensus=json.dumps(consensus), status=consensus["status"])
    )
    return 201, consensus


def my_vote(db, user, sighting_id):
    get_sighting(db, sighting_id)
    vote = db.execute(
        select(VOTES.c.id, VOTES.c.rank, VOTES.c.label)
        .where(VOTES.c.sighting_id == sighting_id, VOTES.c.reviewer_id == user["id"])
        .order_by(VOTES.c.sequence.desc())
        .limit(1)
    ).mappings().one_or_none()
    return {"vote": dict(vote) if vote is not None else None}


def training_review_feed(db, user, environ):
    require_curator(db, user)
    cursor, limit = feed_bounds(environ)
    already_approved = (
        select(APPROVALS.c.sighting_id).where(CURRENT_APPROVAL).correlate(SIGHTINGS).exists()
    )
    rows = db.execute(
        select(SIGHTINGS.c.sequence, SIGHTINGS.c.payload, SIGHTINGS.c.consensus)
        .where(TRAINING_ELIGIBLE, ~already_approved, SIGHTINGS.c.sequence > cursor)
        .order_by(SIGHTINGS.c.sequence)
        .limit(limit + 1)
    ).mappings().all()
    items = [
        {**public_capture(row["payload"]), "consensus": json.loads(row["consensus"])}
        for row in rows[:limit]
    ]
    return page(items, rows, limit)


def approve_for_training(db, user, sighting_id, data):
    require_curator(db, user)
    require_fields(data, ("revision",))
    revision = data["revision"]
    if type(revision) is not int or revision < 1:
        raise RequestError(400, "revision must be a positive integer")
    sighting = get_sighting(db, sighting_id)
    consensus = json.loads(sighting["consensus"])
    if revision != consensus["revision"]:
        raise RequestError(409, "Consensus revision has changed")
    capture = json.loads(sighting["payload"])
    if sighting["status"] not in CONSENSUS_STATUSES or capture.get("training_consent") is not True:
        raise RequestError(409, "Training requires resolved consensus and explicit consent")

    existing = db.execute(
        select(APPROVALS.c.curator_id, APPROVALS.c.approved_at, APPROVALS.c.consensus)
        .where(APPROVALS.c.sighting_id == sighting_id, APPROVALS.c.revision == revision)
    ).mappings().one_or_none()
    if existing is not None:
        return 200, {
            "sighting_id": sighting_id, "revision": revision, "curator_id": existing["curator_id"],
            "approved_at": existing["approved_at"], "consensus": json.loads(existing["consensus"]),
        }

    approval = {
        "sighting_id": sighting_id, "revision": revision, "curator_id": user["id"],
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    db.execute(APPROVALS.insert().values(**approval, consensus=json.dumps(consensus)))
    return 201, {**approval, "consensus": consensus}


def write_training_archive(db, archive):
    records = db.execute(
        select(
            SIGHTINGS.c.id, SIGHTINGS.c.payload, APPROVALS.c.consensus,
            APPROVALS.c.curator_id, APPROVALS.c.approved_at,
        )
        .select_from(SIGHTINGS.join(APPROVALS, CURRENT_APPROVAL))
        .where(TRAINING_ELIGIBLE)
        .order_by(SIGHTINGS.c.sequence)
    ).mappings()
    manifest = []
    for record in records:
        capture = json.loads(record["payload"])
        images = db.execute(
            select(EVIDENCE.c.slot, EVIDENCE.c.jpeg).where(EVIDENCE.c.sighting_id == record["id"])
            .order_by(EVIDENCE.c.slot)
        ).mappings().all()
        if [image["slot"] for image in images] != list(range(len(capture["evidence"]))):
            raise ValueError(f"Missing approved evidence for sighting {record['id']}")
        exported_images = []
        for image in images:
            slot, jpeg = image["slot"], image["jpeg"]
            path = f"images/{record['id']}/{slot}.jpg"
            archive.writestr(path, jpeg)
            exported_images.append({
                "slot": slot, "path": path, "sha256": hashlib.sha256(jpeg).hexdigest(),
            })
        snapshot = json.loads(record["consensus"])
        manifest.append(json.dumps({
            "schema_version": 1,
            "sighting_id": record["id"],
            "consensus": snapshot,
            "approval": {"curator_id": record["curator_id"], "approved_at": record["approved_at"]},
            "capture": capture,
            "images": exported_images,
        }, sort_keys=True))
    archive.writestr("manifest.jsonl", "".join(line + "\n" for line in manifest))
    return len(manifest)


def export_training(path, output, token):
    engine = create_database_engine(path)
    try:
        SCHEMA.create_all(engine)
        with engine.begin() as db:
            # Read approvals, consensus and image bytes from the same database snapshot.
            db.exec_driver_sql("BEGIN")
            user = authenticate(db, {"HTTP_AUTHORIZATION": f"Bearer {token}"})
            require_curator(db, user)
            with output.open("xb") as destination:
                try:
                    with ZipFile(destination, "w") as archive:
                        return write_training_archive(db, archive)
                except Exception:
                    destination.close()
                    output.unlink()
                    raise
    finally:
        engine.dispose()


def dispatch(db, user, environ, data):
    path, method = environ["PATH_INFO"], environ["REQUEST_METHOD"]
    if path == "/sightings" and method == "POST":
        return create_sighting(db, user, data)
    if path == "/sightings/review-feed" and method == "GET":
        return 200, review_feed(db, user, environ)
    if path == "/training/review-feed" and method == "GET":
        return 200, training_review_feed(db, user, environ)

    parts = path.split("/")
    if len(parts) == 4 and parts[1] == "sightings":
        sighting_id = parse_uuid(parts[2], "sighting_id")
        if parts[3] == "votes" and method == "POST":
            return cast_vote(db, user, sighting_id, data)
        if parts[3] == "my-vote" and method == "GET":
            return 200, my_vote(db, user, sighting_id)
        if parts[3] == "consensus" and method == "GET":
            return 200, json.loads(get_sighting(db, sighting_id)["consensus"])
        if parts[3] == "approve-for-training" and method == "POST":
            return approve_for_training(db, user, sighting_id, data)
    if len(parts) == 5 and parts[1] == "sightings" and parts[3] == "evidence":
        sighting_id = parse_uuid(parts[2], "sighting_id")
        if parts[4] not in ("0", "1", "2"):
            raise RequestError(404, "Evidence slot not found")
        slot = int(parts[4])
        if method == "PUT":
            return put_evidence(db, user, sighting_id, slot, data)
        if method == "GET":
            return get_evidence(db, user, sighting_id, slot)
    raise RequestError(404, "Endpoint not found")


def make_app(path):
    engine = create_database_engine(path)
    SCHEMA.create_all(engine)

    def application(environ, start_response):
        try:
            with engine.begin() as db:
                user = authenticate(db, environ)
                data = None
                method = environ["REQUEST_METHOD"]
                if method == "POST":
                    data = read_json(environ)
                elif method == "PUT":
                    data = read_body(environ, "image/jpeg", MAX_EVIDENCE_BYTES)
                if method in ("POST", "PUT"):
                    # Serialize offline replays and vote replacements before reading their state.
                    db.exec_driver_sql("BEGIN IMMEDIATE")
                status, body = dispatch(db, user, environ, data)
        except RequestError as exc:
            status, body = exc.status, {"error": str(exc)}
        status = HTTPStatus(status)
        if isinstance(body, bytes):
            payload, content_type = body, "image/jpeg"
        else:
            payload, content_type = json.dumps(body, allow_nan=False).encode("utf-8"), "application/json"
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(payload))),
            ("Cache-Control", "no-store"),
        ]
        if status == HTTPStatus.UNAUTHORIZED:
            headers.append(("WWW-Authenticate", "Bearer"))
        start_response(f"{status.value} {status.phrase}", headers)
        return [payload]

    return application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    user_parser = commands.add_parser("add-user")
    user_parser.add_argument("role", choices=ROLE_WEIGHT)
    user_parser.add_argument("--training-curator", action="store_true")
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8080)
    export_parser = commands.add_parser("export-training")
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "add-user":
        print(json.dumps(provision_user(args.db, args.role, training_curator=args.training_curator)))
    elif args.command == "export-training":
        try:
            count = export_training(args.db, args.output, getpass("Curator token: "))
        except (RequestError, OSError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Exported {count} approved sightings to {args.output}")
    else:
        with make_server("127.0.0.1", args.port, make_app(args.db)) as server:
            print(f"Listening on http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    main()
