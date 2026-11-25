import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../models/records.dart';
import '../models/verdict.dart';
import '../services/sos_service.dart';
import '../state/app_state.dart';
import '../util/geo.dart';
import '../widgets/common.dart';

/// The emergency screen. Reached from "I've been bitten", the SOS button on
/// every tab, and the result screen. Every action opens the phone's own
/// dialler, SMS app or maps: the user presses the final button.
class SosScreen extends StatelessWidget {
  const SosScreen({super.key, this.bitten = false, this.identification});
  final bool bitten;
  final Identification? identification;

  Future<void> _textContacts(BuildContext context, AppState s) async {
    final contacts = s.emergencyContacts;
    final messenger = ScaffoldMessenger.of(context);
    if (contacts.isEmpty) {
      messenger.showSnackBar(
        const SnackBar(
          content: Text('Add an emergency contact in the Contacts tab first.'),
        ),
      );
      return;
    }
    final msg = SosService.message(
      bitten: bitten,
      location: s.position,
      accuracy: s.accuracy,
      address: s.address,
      species: identification?.isDemo == true
          ? null
          : identification?.topSpecies,
      verdict: identification?.isDemo == true ? null : identification?.verdict,
    );
    final opened = await s.sos.textContacts(contacts, msg);
    if (!opened) {
      messenger.showSnackBar(
        const SnackBar(content: Text('Could not open the SMS app.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final hospitals = s.hospitalsNearby.take(8).toList();
    final species = identification?.isDemo == true
        ? null
        : identification?.topSpecies;

    return Scaffold(
      appBar: AppBar(
        title: Text(bitten ? 'Snakebite emergency' : 'Emergency help'),
        backgroundColor: bitten ? const Color(0xFFB3261E) : null,
        foregroundColor: bitten ? Colors.white : null,
        titleTextStyle: bitten
            ? const TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.w800,
              )
            : null,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
          children: [
            if (bitten)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: scheme.errorContainer,
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Go to a hospital now',
                      style: text.titleLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: scheme.onErrorContainer,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Call an ambulance, keep as still and calm as you can, and get to a '
                      'hospital that has anti-snake venom (ASV). Do this even if the snake '
                      'looked harmless.',
                      style: text.bodyLarge?.copyWith(
                        color: scheme.onErrorContainer,
                      ),
                    ),
                    if (species != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        'Tell the doctor: possibly ${species.fullName}. Show them the photo.',
                        style: text.bodyMedium?.copyWith(
                          color: scheme.onErrorContainer,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            const SectionTitle('Call'),
            ...nationalNumbers.map(
              (n) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: n.number == '102'
                    ? _CallTile(number: n, onTap: () => s.sos.call(n.number))
                    : EmergencyButton(
                        label: 'Call ${n.number} · ${n.label}',
                        icon: n.number == '108'
                            ? Icons.airport_shuttle_rounded
                            : Icons.phone_in_talk_rounded,
                        onPressed: () => s.sos.call(n.number),
                      ),
              ),
            ),
            const SectionTitle('Tell your people'),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (s.position == null)
                      Text('Location not found yet. The message will ask them to call you.',
                          style: text.bodyMedium)
                    else ...[
                      Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Icon(Icons.place_rounded, size: 20, color: scheme.primary),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(s.address ?? 'Finding your address…',
                              style: text.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
                        ),
                      ]),
                      Padding(
                        padding: const EdgeInsets.only(left: 26, top: 2),
                        child: Text(
                          '${s.position}${s.accuracy == null ? '' : ' · ±${s.accuracy!.round()} m'}',
                          style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                        ),
                      ),
                    ],
                    const SizedBox(height: 4),
                    Text(
                      s.emergencyContacts.isEmpty
                          ? 'No emergency contacts saved yet.'
                          : 'Sends to: ${s.emergencyContacts.map((c) => c.name).join(', ')}',
                      style: text.bodySmall?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: FilledButton.icon(
                        onPressed: () => _textContacts(context, s),
                        icon: const Icon(Icons.sms_rounded),
                        label: const Text(
                          'Text my location to emergency contacts',
                        ),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Needs mobile signal; mobile data is not needed.',
                      style: text.bodySmall?.copyWith(
                        color: scheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            ...s.contacts
                .where((c) => c.kind != ContactKind.emergency)
                .map(
                  (c) => Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Card(
                      child: ListTile(
                        leading: const Icon(Icons.support_agent_rounded),
                        title: Text(c.name),
                        subtitle: Text(c.kind.label),
                        trailing: IconButton.filledTonal(
                          onPressed: () => s.sos.call(c.phone),
                          icon: const Icon(Icons.call_rounded),
                          tooltip: 'Call ${c.name}',
                        ),
                      ),
                    ),
                  ),
                ),
            SectionTitle(
              'Nearest hospitals',
              trailing: s.hospitalsLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : IconButton(
                      tooltip: 'Refresh hospitals',
                      onPressed: s.refreshHospitals,
                      icon: const Icon(Icons.refresh_rounded),
                    ),
            ),
            if (s.hospitalsError != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  s.hospitalsError!,
                  style: TextStyle(color: scheme.error),
                ),
              ),
            if (hospitals.isEmpty && !s.hospitalsLoading)
              Text(
                s.position == null
                    ? 'Finding your location. Nearby hospitals appear once the phone knows where you are.'
                    : 'No hospitals saved for this area yet. Connect to the internet once and tap refresh; '
                        'the list is then kept on the phone for offline use.',
                style: text.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ...hospitals.map((h) => _HospitalCard(hospital: h)),
            if (hospitals.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Hospital data © OpenStreetMap contributors'
                  '${s.hospitalsFetched == null ? '' : ', saved ${formatAge(DateTime.now().difference(s.hospitalsFetched!))}'}. '
                  'Not every hospital stocks ASV: call ahead if you can.',
                  style: text.bodySmall?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _CallTile extends StatelessWidget {
  const _CallTile({required this.number, required this.onTap});
  final EmergencyNumber number;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Card(
    child: ListTile(
      leading: const Icon(Icons.airport_shuttle_outlined),
      title: Text('${number.number} · ${number.label}'),
      subtitle: Text(number.detail),
      trailing: IconButton.filledTonal(
        onPressed: onTap,
        icon: const Icon(Icons.call_rounded),
        tooltip: 'Call ${number.number}',
      ),
    ),
  );
}

class _HospitalCard extends StatelessWidget {
  const _HospitalCard({required this.hospital});
  final Hospital hospital;

  @override
  Widget build(BuildContext context) {
    final s = context.read<AppState>();
    final scheme = Theme.of(context).colorScheme;
    final here = s.position;
    final dist = here == null ? null : distanceMeters(here, hospital.location);
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 8, 12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    hospital.name,
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 15,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    [
                      if (dist != null)
                        '${formatDistance(dist)} ${compassPoint(bearingDegrees(here!, hospital.location))} (straight line)',
                      if (hospital.emergency == true) 'Emergency department',
                      if (hospital.isSpecialtyOnly)
                        'Specialty: ${hospital.speciality!.replaceAll(';', ', ')}',
                      if (hospital.address != null) hospital.address!,
                    ].join(' · '),
                    style: TextStyle(
                      color: scheme.onSurfaceVariant,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),
            if (hospital.phone != null)
              IconButton(
                tooltip: 'Call ${hospital.name}',
                onPressed: () => s.sos.call(hospital.phone!),
                icon: const Icon(Icons.call_rounded),
              ),
            IconButton(
              tooltip: 'Directions to ${hospital.name}',
              onPressed: () => s.sos.directions(hospital.location),
              icon: const Icon(Icons.directions_rounded),
            ),
          ],
        ),
      ),
    );
  }
}
