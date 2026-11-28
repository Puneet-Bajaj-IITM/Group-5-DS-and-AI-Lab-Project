using System;
using System.Collections.Generic;
using System.Threading;
using Windows.Devices.Bluetooth.Advertisement;
using Windows.Storage.Streams;
public static class SerpentBleReceiver {
  private const string Expected = "535401B0280001FF020000000000000000000000";
  public static string Listen(int seconds) {
    if (seconds < 1 || seconds > 60) throw new ArgumentOutOfRangeException("seconds");
    var records = new List<string>();
    var candidates = new HashSet<string>();
    int totalEvents = 0;
    var gate = new object();
    var watcher = new BluetoothLEAdvertisementWatcher();
    watcher.ScanningMode = BluetoothLEScanningMode.Active;
    watcher.AllowExtendedAdvertisements = true;
    watcher.Received += (sender, args) => {
      Interlocked.Increment(ref totalEvents);
      foreach (var manufacturer in args.Advertisement.ManufacturerData) {
        if (manufacturer.Data.Length < 2 || manufacturer.Data.Length > 64) continue;
        byte[] bytes = new byte[manufacturer.Data.Length];
        using (var reader = DataReader.FromBuffer(manufacturer.Data)) { reader.ReadBytes(bytes); }
        string hex = BitConverter.ToString(bytes).Replace("-", "");
        if (bytes[0] == 0x53 && bytes[1] == 0x54) { lock(gate) { if(candidates.Count < 8) candidates.Add("{\"company_id\":\"0x" + manufacturer.CompanyId.ToString("X4") + "\",\"bytes\":" + bytes.Length + ",\"hex\":\"" + hex + "\"}"); } }
        if (manufacturer.CompanyId != 0xFFFF || hex != Expected) continue;
        string row = "{\"utc\":\"" + DateTime.UtcNow.ToString("o") + "\",\"rssi_dbm\":" + args.RawSignalStrengthInDBm + ",\"payload_hex\":\"" + hex + "\"}";
        lock(gate) { records.Add(row); }
      }
    };
    try {
      watcher.Start();
      Console.WriteLine("BLE_RECEIVER_READY: only the exact ST test payload will be recorded");
      Thread.Sleep(seconds * 1000);
      lock(gate) {
        return "{\"scope\":\"phone-to-Windows isolated BLE radio probe; not SerpentGuard end-to-end or two-phone delivery\",\"watcher_status\":\"" + watcher.Status + "\",\"seconds\":" + seconds + ",\"matching_reports\":" + records.Count + ",\"total_advertisement_events\":" + totalEvents + ",\"test_prefix_candidates\":[" + String.Join(",",new List<string>(candidates).ToArray()) + "],\"records\":[" + String.Join(",",records.ToArray()) + "]}";
      }
    } finally { watcher.Stop(); }
  }
}
