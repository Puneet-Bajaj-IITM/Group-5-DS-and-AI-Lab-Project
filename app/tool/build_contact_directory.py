"""Build the small, official-source-checked offline contact directory.

Source pages were read on 2026-11-25. No telephone verification was performed.
Edit this reviewed source, then run from any directory with the existing venv.
"""
import json
from pathlib import Path

CHECKED = "2026-11-25"
SOS = "https://wildlifesos.org/contact-us/"
RESQ = "https://www.resqct.org/contact"
VET = "https://www.tanuvas.ac.in/contacts.php"


def record(identifier, name, phone, kind, locality, state, source, notes, **extra):
    return dict(id=identifier, name=name, phone=phone, kind=kind,
                locality=locality, state=state, source_url=source,
                source_checked_at=CHECKED,
                verification_method="official_website_not_called",
                notes=notes, location=None, **extra)


records = [
    record("ws-delhi", "Wildlife SOS Delhi helpline", "+919871963535", "rescuer",
           "Delhi NCT", "Delhi", SOS, "Regional wildlife rescue helpline; call to confirm coverage."),
    record("ws-agra", "Wildlife SOS Agra helpline", "+919917109666", "rescuer",
           "Agra", "Uttar Pradesh", SOS, "Regional wildlife rescue helpline; call to confirm coverage."),
    record("ws-vadodara", "Wildlife SOS Vadodara helpline", "+919825011117", "rescuer",
           "Vadodara", "Gujarat", SOS, "Regional wildlife rescue helpline; call to confirm coverage."),
    record("ws-srinagar", "Wildlife SOS Srinagar helpline", "+917006692300", "rescuer",
           "Srinagar", "Jammu and Kashmir", SOS, "Regional wildlife rescue helpline; call to confirm coverage."),
    record("resq-pune", "RESQ wildlife helpline Pune", "+919172511100", "rescuer",
           "Bavdhan Pune", "Maharashtra", RESQ,
           "Wildlife emergencies and technical animal rescue. Distance is to the RESQ centre, not a rescuer's live position."),
    record("resq-baramati", "RESQ wildlife helpline Baramati", "+917387771050", "rescuer",
           "Baramati", "Maharashtra", RESQ, "Wildlife emergencies; call to confirm response area."),
    record("resq-roha", "RESQ wildlife helpline Roha", "+919922330286", "rescuer",
           "Roha", "Maharashtra", RESQ, "Wildlife emergencies; call to confirm response area."),
    record("resq-nashik", "RESQ wildlife helpline Nashik", "+918698112211", "rescuer",
           "Nashik", "Maharashtra", RESQ, "Wildlife emergencies; call to confirm response area."),
    record("mvc-chennai", "Madras Veterinary College Teaching Hospital", "+914425381366",
           "veterinarian", "Vepery Chennai", "Tamil Nadu", VET,
           "Animal treatment contact, not a human snakebite hospital. Call for opening hours and species capability."),
    record("vuph-chennai", "Veterinary University Peripheral Hospital", "+914425559046",
           "veterinarian", "Madhavaram Milk Colony Chennai", "Tamil Nadu", VET,
           "Animal treatment contact, not a human snakebite hospital. Call for opening hours and species capability."),
]
# Coordinates come from the map embedded on RESQ's own contact page. Regional
# helplines and unmapped veterinary facilities deliberately have no point.
records[4]["location"] = {"lat": 18.4949046, "lon": 73.7652262}
records[4]["coordinate_source_url"] = RESQ
records[4]["coordinate_method"] = "official_contact_page_google_maps_embed"
target = Path(__file__).resolve().parents[1] / "assets/contact_directory.json"
target.write_text(json.dumps(dict(schema_version=1, revision=20261125,
                                 coverage="Pilot directory; not nationwide coverage",
                                 contacts=records), ensure_ascii=False, indent=2) + "\n")
print(f"Wrote {len(records)} official-source-checked contacts; 1 mapped centre; no phone verification")
