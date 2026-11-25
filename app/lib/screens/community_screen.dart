import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../services/voting_api.dart';
import '../services/voting_queue.dart';
import '../services/voting_background.dart';
import '../services/voting_session.dart';
import '../state/app_state.dart';
import 'live_review_screen.dart';

/// The server is authoritative for roles and consensus. Offline actions are
/// pending until acknowledged; a locally queued vote never becomes a label.
class CommunityScreen extends StatefulWidget {
  const CommunityScreen({super.key, this.openQueue, this.initialEvidence});
  final Future<VotingQueue> Function()? openQueue;
  final LiveEvidence? initialEvidence;
  @override
  State<CommunityScreen> createState() => _CommunityScreenState();
}

class _CommunityScreenState extends State<CommunityScreen>
    with WidgetsBindingObserver {
  VotingQueue? _queue;
  VotingApi? _api;
  final _server = TextEditingController(text: 'http://10.0.2.2:8080');
  final _token = TextEditingController();
  List<Map<String, Object?>> _jobs = [];
  List<Map<String, dynamic>> _feed = [];
  bool _busy = false;
  bool _remember = true;
  int? _nextCursor;
  final _sessions = VotingSessionStore();
  String _status =
      'Capture works offline. Save a draft now; connect an account to submit it or review the community feed.';
  Timer? _timer;
  bool _initialShared = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _open();
    // Foreground updates complement Android's remembered-account worker.
    _timer = Timer.periodic(const Duration(minutes: 1), (_) {
      if (_api != null &&
          !_busy &&
          WidgetsBinding.instance.lifecycleState == AppLifecycleState.resumed) {
        unawaited(_sync());
      }
    });
  }

  Future<void> _open() async {
    try {
      final queue = await (widget.openQueue?.call() ?? VotingQueue.open());
      if (!mounted) {
        await queue.db.close();
        return;
      }
      setState(() => _queue = queue);
      try {
        final saved = await _sessions.load();
        if (saved != null && mounted) {
          _api = saved.createApi();
          _server.text = saved.server;
          await _reload();
          if (widget.initialEvidence == null) await _sync();
        }
      } catch (_) {
        if (mounted) {
          setState(
            () => _status = 'Saved account unavailable. Enter your token again; queued actions are kept.',
          );
        }
      }
      await _reload();
    } catch (_) {
      if (mounted) {
        setState(
          () => _status =
              'Could not open offline storage. No submission was saved.',
        );
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && _api != null && !_busy) {
      unawaited(_sync());
    }
  }

  Future<void> _reload() async {
    final api = _api;
    if (_queue == null) return;
    final drafts = await _queue!.jobs(VotingQueue.draftAccount);
    final jobs = api == null ? drafts : [...drafts, ...await _queue!.jobs(api.account)];
    final feed = api == null ? <Map<String, dynamic>>[] : await _queue!.feed(api.account);
    final next = api == null ? null : await _queue!.nextCursor(api.account);
    if (mounted) {
      setState(() {
        _jobs = jobs;
        _feed = feed;
        _nextCursor = next;
      });
    }
  }

  Future<void> _attachDrafts() async {
    final api = _api;
    if (api == null) return;
    final count = _jobs.where((j) => j['account'] == VotingQueue.draftAccount).length;
    final ok = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
      title: const Text('Submit saved drafts with this account?'),
      content: Text('Attach $count saved encounter(s) to this account at ${api.base}? '
          'They may upload when a connection is available, using the sharing consent saved with each encounter.'),
      actions: [TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Keep local')),
        FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Attach and queue'))],
    ));
    if (ok == true) {
      await _queue!.claimDrafts(api.account);
      await _schedule();
      await _reload();
    }
  }

  Future<void> _shareInitial() async {
    final evidence = widget.initialEvidence!;
    try {
      final saved = await _share(evidence.jpegs, evidence.started, offsets: evidence.offsets);
      if (saved && mounted) setState(() => _initialShared = true);
    } catch (_) {
      if (mounted) setState(() => _status = 'Could not save these views. Choose distinct stills and try again.');
    }
  }

  Future<void> _connect() async {
    try {
      final api = VotingApi(_server.text, _token.text);
      if (_remember) {
        final session = VotingSession(api.base.toString(), _token.text.trim());
        await _sessions.save(session);
        try {
          await VotingBackground.schedule(session);
        } catch (_) {
          /* Foreground sync remains available. */
        }
      } else {
        await _sessions.clear();
        await VotingBackground.cancel();
      }
      _api?.close();
      _api = api;
      _token.clear();
      await _reload();
      await _sync(refresh: true);
    } catch (e) {
      if (mounted) setState(() => _status = '$e');
    }
  }

  Future<void> _sync({bool refresh = false, bool more = false}) async {
    final api = _api;
    if (_busy || api == null || _queue == null) return;
    setState(() {
      _busy = true;
      _status = 'Syncing saved actions…';
    });
    try {
      await _queue!.sync(
        api,
        allowed: () async => mounted && _api?.account == api.account,
      );
      if (refresh) {
        await _queue!.refresh(api, more: more);
      }
      await _reload();
      if (mounted) {
        setState(
          () => _status = refresh
              ? 'Review page saved. Cached evidence works offline; refresh for current server status.'
              : 'Sync checked. Saved actions below show server acknowledgement or retry status.',
        );
      }
      await _reload();
    } catch (_) {
      if (mounted) {
        setState(
          () => _status = 'Server unavailable. Showing saved data; pending actions remain on this phone.',
        );
      }
      await _reload();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _schedule() async {
    if (!_remember || _api == null) return;
    try {
      final saved = await _sessions.load();
      if (saved?.account == _api!.account) {
        await VotingBackground.schedule(saved!);
      }
    } catch (_) {
      if (mounted) {
        setState(
          () => _status =
              'Saved locally. Background scheduling failed; use Sync queue.',
        );
      }
    }
  }

  Future<void> _disconnect() async {
    setState(() => _busy = true);
    try {
      await _sessions.clear();
      await VotingBackground.cancel();
      _api?.close();
      if (mounted) {
        setState(() {
          _api = null;
          _jobs = [];
          _feed = [];
          _nextCursor = null;
          _status = 'Account disconnected. Saved actions remain isolated on this phone.';
        });
      }
      await _reload();
    } catch (_) {
      if (mounted) {
        setState(
          () => _status = 'Could not finish disconnecting. Try again before switching accounts.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _photo(ImageSource source) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      final file = await ImagePicker().pickImage(
        source: source,
        maxWidth: 2048,
        maxHeight: 2048,
        imageQuality: 95,
      );
      if (file == null || !mounted) return;
      final jpeg = await compute(votingJpeg, await file.readAsBytes());
      if (mounted) await _share([jpeg], DateTime.now());
    } catch (_) {
      if (mounted) {
        setState(
          () => _status = 'Could not prepare this photo. Nothing was queued.',
        );
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _live() async {
    final result = await Navigator.of(context).push<LiveEvidence>(
      MaterialPageRoute(builder: (_) => const LiveReviewScreen()),
    );
    if (result != null && mounted) {
      try {
        await _share(result.jpegs, result.started, offsets: result.offsets);
      } catch (_) {
        if (mounted) {
          setState(
            () => _status = 'Could not save these stills. Choose distinct views and try again.',
          );
        }
      }
    }
  }

  Future<bool> _share(
    List<Uint8List> jpegs,
    DateTime captured, {
    List<int>? offsets,
  }) async {
    final api = _api;
    if (_queue == null) return false;
    var sameSnake = false;
    var consent = false;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, update) => AlertDialog(
          title: const Text('Share for community identification?'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                for (final jpeg in jpegs)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: Image.memory(jpeg, height: 140, fit: BoxFit.contain),
                  ),
                const Text(
                  'These photos will be visible to authenticated reviewers when uploaded. '
                  'Check for people or identifying details. GPS metadata is removed. '
                  'No demo predictions or precise location will be uploaded.',
                ),
                CheckboxListTile(
                  value: sameSnake,
                  onChanged: (v) => update(() => sameSnake = v ?? false),
                  title: const Text(
                    'I can see one snake, it is unidentified, and all views show the same encounter.',
                  ),
                ),
                CheckboxListTile(
                  value: consent,
                  onChanged: (v) => update(() => consent = v ?? false),
                  title: const Text(
                    'Also allow these images to be reviewed for model improvement (optional).',
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: sameSnake ? () => Navigator.pop(ctx, true) : null,
              child: const Text('Save to offline queue'),
            ),
          ],
        ),
      ),
    );
    if (confirmed != true) return false;
    await _queue!.sighting(
      api?.account ?? VotingQueue.draftAccount,
      jpegs,
      trainingConsent: consent,
      capturedAt: captured,
      offsets: offsets,
    );
    await _schedule();
    await _reload();
    if (mounted) {
      setState(
        () => _status = api == null ? 'Draft saved on this phone. Attach it to an account when ready to submit.'
            : 'Saved on this phone. Sync when internet is available.',
      );
    }
    return true;
  }

  Future<void> _review(Map<String, dynamic> item) async {
    final api = _api!;
    final images = await _queue!.images(api.account, item['id'] as String);
    final expected = (item['evidence'] as List? ?? []).length;
    if (!mounted) return;
    if (images.isEmpty || images.length != expected) {
      setState(
        () => _status = 'Evidence is not fully cached. Refresh online before reviewing this encounter.',
      );
      return;
    }
    final catalog = context.read<AppState>().catalog;
    var rank = 'species';
    String? label;
    List<String> labels() =>
        (rank == 'species'
              ? catalog.all.map((s) => s.name).toSet().toList()
              : catalog.all.map((s) => s.genus).toSet().toList())
          ..sort();
    final vote = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, update) => AlertDialog(
          title: const Text('Identify this encounter'),
          content: SizedBox(
            width: 400,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  for (final jpeg in images)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Image.memory(
                        jpeg,
                        height: 180,
                        fit: BoxFit.contain,
                      ),
                    ),
                  const Text(
                    'Skip if these views do not support an identification. '
                    'A community vote does not change emergency advice.',
                  ),
                  DropdownButton<String>(
                    value: rank,
                    isExpanded: true,
                    items: const [
                      DropdownMenuItem(
                        value: 'species',
                        child: Text('Species'),
                      ),
                      DropdownMenuItem(value: 'genus', child: Text('Genus')),
                    ],
                    onChanged: (v) => update(() {
                      rank = v!;
                      label = null;
                    }),
                  ),
                  DropdownButton<String>(
                    value: label,
                    hint: const Text('Choose an identification'),
                    isExpanded: true,
                    items: labels()
                        .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                        .toList(),
                    onChanged: (v) => update(() => label = v),
                  ),
                  if (item['your_vote'] != null)
                    Text('Your previous vote: ${item['your_vote']['label']}'),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Skip'),
            ),
            FilledButton(
              onPressed: label == null ? null : () => Navigator.pop(ctx, true),
              child: const Text('Queue vote'),
            ),
          ],
        ),
      ),
    );
    if (vote == true) {
      try {
        await _queue!.vote(
          api.account,
          item['id'] as String,
          rank,
          label!,
          previous: (item['your_vote'] as Map?)?['id'] as String?,
        );
        await _schedule();
        await _reload();
        if (mounted) {
          setState(
            () => _status =
                'Vote saved locally. Sync to submit it to community consensus.',
          );
        }
      } catch (_) {
        if (mounted) {
          setState(
            () => _status = 'There is already an unsynced vote for this sighting. Check the queue.',
          );
        }
      }
    }
  }

  Future<void> _reconcile(Map<String, Object?> job) async {
    try {
      final current = await _api!.json(
        'GET',
        '/sightings/${job['sighting']}/my-vote',
      );
      if (!mounted) return;
      final vote = current['vote'] as Map?;
      final ok = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Reconcile queued vote'),
          content: Text(
            'Current server vote: ${vote?['label'] ?? 'none'}. '
            'Queue your saved identification as a replacement? It will still need to sync.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Keep conflict'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Confirm replacement'),
            ),
          ],
        ),
      );
      if (ok == true) {
        await _queue!.reconcile(
          job['id'] as String,
          _api!.account,
          vote?['id'] as String?,
        );
        await _schedule();
        await _reload();
      }
    } catch (_) {
      if (mounted) {
        setState(
          () => _status = 'Could not reconcile. The saved vote has been kept.',
        );
      }
    }
  }

  Future<void> _consensus(String id) async {
    final result = await _queue!.consensus(_api!, id);
    if (!mounted) return;
    final value = result.value;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(
          result.cached
              ? 'Saved community result (may be stale)'
              : 'Community result',
        ),
        content: Text(
          value == null
              ? 'No result is cached. Connect to fetch the server result.'
              : '${value['status']}\n${value['rank'] ?? 'No rank'}: ${value['label'] ?? 'Unresolved'}\n'
                    'Posterior: ${value['posterior_confidence'] ?? 'pending'}\n'
                    'Reviewers: ${value['reviewer_count']} · Revision: ${value['revision']}\n\n'
                    'Community identification is not medical advice and does not change emergency guidance.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Community review')),
    body: ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          context.read<AppState>().classifier.isDemo
              ? 'Identification is currently a demo. Community sharing below uses '
                    'your confirmed snake evidence, without simulated model predictions.'
              : 'Community sharing below uses your own photos of the snake. '
                    'Reviewers decide the species themselves; the model\'s answer is not sent.',
        ),
        const SizedBox(height: 12),
        if (_api == null) ...[
          TextField(
            controller: _server,
            decoration: const InputDecoration(
              labelText: 'Community server origin',
            ),
          ),
          TextField(
            controller: _token,
            obscureText: true,
            enableSuggestions: false,
            autocorrect: false,
            decoration: const InputDecoration(
              labelText: 'Provisioned account token',
            ),
          ),
          CheckboxListTile(
            value: _remember,
            onChanged: (v) => setState(() => _remember = v ?? false),
            title: const Text(
              'Remember account securely and sync queued actions in the background on Android',
            ),
          ),
          FilledButton(
            onPressed: _queue == null || _busy ? null : _connect,
            child: const Text('Use account'),
          ),
          const Text(
            'Session-only mode requires your token again after restart. '
            'HTTP works only with localhost / Android emulator for development.',
          ),
          Wrap(spacing: 8, children: [
            OutlinedButton(onPressed: _busy || _queue == null ? null : () => _photo(ImageSource.camera), child: const Text('Save photo draft')),
            OutlinedButton(onPressed: _busy || _queue == null ? null : () => _photo(ImageSource.gallery), child: const Text('Gallery draft')),
            OutlinedButton(onPressed: _busy || _queue == null ? null : _live, child: const Text('Live camera')),
          ]),
          if (widget.initialEvidence != null && !_initialShared)
            FilledButton(onPressed: _busy || _queue == null ? null : _shareInitial,
                child: const Text('Review and save selected live-camera views')),
        ] else ...[
          Text('Server: ${_api!.base}'),
          if (_jobs.any((j) => j['account'] == VotingQueue.draftAccount))
            FilledButton(onPressed: _busy ? null : _attachDrafts,
                child: const Text('Attach local drafts to this account')),
          TextButton(
            onPressed: _busy ? null : _disconnect,
            child: const Text('Disconnect account (keeps saved queue)'),
          ),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : () => _photo(ImageSource.camera),
                icon: const Icon(Icons.photo_camera),
                label: const Text('Share photo'),
              ),
              OutlinedButton(
                onPressed: _busy ? null : () => _photo(ImageSource.gallery),
                child: const Text('Gallery'),
              ),
              OutlinedButton(
                onPressed: _busy ? null : _live,
                child: const Text('Live camera stills'),
              ),
              OutlinedButton(
                onPressed: _busy ? null : () => _sync(),
                child: const Text('Sync queue'),
              ),
              OutlinedButton(
                onPressed: _busy ? null : () => _sync(refresh: true),
                child: const Text('Refresh review feed'),
              ),
            ],
          ),
          if (widget.initialEvidence != null && !_initialShared)
            FilledButton.icon(
              icon: const Icon(Icons.collections),
              label: Text(
                'Share ${widget.initialEvidence!.jpegs.length} selected live-camera views',
              ),
              onPressed: _busy
                  ? null
                  : () async {
                      final evidence = widget.initialEvidence!;
                      try {
                        final saved = await _share(
                          evidence.jpegs,
                          evidence.started,
                          offsets: evidence.offsets,
                        );
                        if (saved && mounted) {
                          setState(() => _initialShared = true);
                        }
                      } catch (_) {
                        if (mounted) {
                          setState(
                            () => _status = 'Could not save these views. Check that they are distinct.',
                          );
                        }
                      }
                    },
            ),
        ],
        const SizedBox(height: 12),
        if (_busy) const LinearProgressIndicator(),
        Text(_status, semanticsLabel: _status),
        const SizedBox(height: 16),
        Text(
          'Saved actions (${_jobs.where((j) => j['state'] != 'done').length} unsynced)',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        for (final job in _jobs.take(30))
          ListTile(
            title: Text('${job['kind']} · ${job['account'] == VotingQueue.draftAccount ? 'local draft' : job['state']}'),
            subtitle: Text(
              '${job['sighting']}${job['error'] == null ? '' : '\n${job['error']}'}',
            ),
            trailing: job['state'] == 'done'
                ? TextButton(
                    onPressed: _busy
                        ? null
                        : () => _consensus(job['sighting'] as String),
                    child: const Text('Result'),
                  )
                : job['state'] == 'blocked' && job['kind'] == 'vote'
                ? TextButton(
                    onPressed: _busy ? null : () => _reconcile(job),
                    child: const Text('Reconcile'),
                  )
                : null,
          ),
        if (_jobs.any(
          (j) => j['state'] == 'blocked' && j['kind'] == 'sighting',
        ))
          TextButton(
            onPressed: _busy
                ? null
                : () async {
                    await _queue!.retry(_api!.account);
                    await _sync();
                  },
            child: const Text('Retry blocked uploads with the same request'),
          ),
        Text(
          'Unresolved sightings (cached)',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        for (final item in _feed)
          Card(
            child: ListTile(
              title: Text('Encounter · ${item['source'] ?? 'photo'}'),
              subtitle: Text(
                '${item['captured_at']}\n${item['your_vote'] == null ? 'No vote submitted' : 'Your vote: ${item['your_vote']['label']}'}',
              ),
              trailing: const Icon(Icons.chevron_right),
              onTap: _busy ? null : () => _review(item),
            ),
          ),
        if (_nextCursor != null)
          TextButton(
            onPressed: _busy ? null : () => _sync(refresh: true, more: true),
            child: const Text('Load more encounters'),
          ),
      ],
    ),
  );

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer?.cancel();
    _api?.close();
    // Outstanding operations may still hold the database; sqflite's app-wide
    // connection stays open until process shutdown rather than closing mid-sync.
    _server.dispose();
    _token.dispose();
    super.dispose();
  }
}
