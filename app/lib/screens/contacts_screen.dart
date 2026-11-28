import 'package:material_ui/material_ui.dart';
import 'package:provider/provider.dart';

import '../models/records.dart';
import '../services/sos_service.dart';
import '../state/app_state.dart';
import '../widgets/common.dart';
import '../widgets/contact_directory_panel.dart';

class ContactsScreen extends StatelessWidget {
  const ContactsScreen({super.key});

  Future<void> _add(BuildContext context) async {
    final c = await showModalBottomSheet<Contact>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => const _ContactForm(),
    );
    if (c != null && context.mounted) await context.read<AppState>().addContact(c);
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;

    Widget group(ContactKind kind, String title, String empty) {
      final list = s.contacts.where((c) => c.kind == kind).toList();
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SectionTitle(title),
        if (list.isEmpty)
          Text(empty, style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant)),
        ...list.map((c) => Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ListTile(
                title: Text(c.name, style: const TextStyle(fontWeight: FontWeight.w600)),
                subtitle: Text(c.phone),
                trailing: Row(mainAxisSize: MainAxisSize.min, children: [
                  IconButton(
                    tooltip: 'Call ${c.name}',
                    onPressed: () => s.sos.call(c.phone),
                    icon: const Icon(Icons.call_rounded),
                  ),
                  IconButton(
                    tooltip: 'Remove ${c.name}',
                    onPressed: () => s.removeContact(c),
                    icon: const Icon(Icons.delete_outline_rounded),
                  ),
                ]),
              ),
            )),
      ]);
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Contacts')),
      floatingActionButton: FloatingActionButton.extended(
        heroTag: 'add-contact',
        onPressed: () => _add(context),
        icon: const Icon(Icons.person_add_alt_1_rounded),
        label: const Text('Add contact'),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.startFloat,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 160),
          children: [
            const ContactDirectoryPanel(),
            group(ContactKind.emergency, 'Emergency contacts',
                'Family or friends who get your location by SMS when you press SOS.'),
            group(ContactKind.rescuer, 'Snake rescuers',
                'A local snake rescuer can move a snake safely. Ask your panchayat or forest office.'),
            group(ContactKind.forest, 'Forest and wildlife offices', 'Your district forest or wildlife office.'),
            group(ContactKind.doctor, 'Doctors and clinics', 'A nearby doctor or clinic.'),
            group(ContactKind.veterinarian, 'Veterinarians', 'Animal hospitals and vets you saved.'),
            const SectionTitle('National numbers'),
            ...nationalNumbers.map((n) => Card(
                  margin: const EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    leading: CircleAvatar(child: Text(n.number, style: const TextStyle(fontSize: 13))),
                    title: Text(n.label),
                    subtitle: Text(n.detail),
                    trailing: IconButton(
                      tooltip: 'Call ${n.number}',
                      onPressed: () => s.sos.call(n.number),
                      icon: const Icon(Icons.call_rounded),
                    ),
                  ),
                )),
            const SizedBox(height: 4),
            Text('Contacts are stored on this phone only.',
                style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}

class _ContactForm extends StatefulWidget {
  const _ContactForm();

  @override
  State<_ContactForm> createState() => _ContactFormState();
}

class _ContactFormState extends State<_ContactForm> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _phone = TextEditingController();
  ContactKind _kind = ContactKind.emergency;

  @override
  void dispose() {
    _name.dispose();
    _phone.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.fromLTRB(16, 0, 16, 16 + MediaQuery.viewInsetsOf(context).bottom),
      child: Form(
        key: _form,
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('Add contact', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 16),
          DropdownButtonFormField<ContactKind>(
            initialValue: _kind,
            decoration: const InputDecoration(labelText: 'Contact type'),
            items: [for (final kind in ContactKind.values)
              DropdownMenuItem(value: kind, child: Text(kind.label))],
            onChanged: (v) { if (v != null) setState(() => _kind = v); },
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _name,
            textCapitalization: TextCapitalization.words,
            decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
            validator: (v) => (v == null || v.trim().isEmpty) ? 'Enter a name' : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _phone,
            keyboardType: TextInputType.phone,
            decoration: const InputDecoration(
                labelText: 'Phone number', hintText: '+91 98765 43210', border: OutlineInputBorder()),
            validator: (v) {
              final digits = (v ?? '').replaceAll(RegExp(r'[^0-9]'), '');
              return digits.length < 3 ? 'Enter a phone number' : null;
            },
          ),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: () {
              if (!_form.currentState!.validate()) return;
              Navigator.pop(
                context,
                Contact(name: _name.text.trim(), phone: _phone.text.trim(), kind: _kind),
              );
            },
            child: const Padding(padding: EdgeInsets.all(12), child: Text('Save')),
          ),
        ]),
      ),
    );
  }
}
