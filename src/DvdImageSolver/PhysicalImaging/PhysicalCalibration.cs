using System.Globalization;
using System.Text.Json;
using DiscImageStudio.ImageIO;

namespace DvdImageSolver.PhysicalImaging;

public static class PhysicalCalibration
{
    public static void Write(string outputDirectory, int size = 1201, double radius = 58)
    {
        if (size is < 33 or > 8192 || !double.IsFinite(radius) || radius <= 25)
            throw new ArgumentException("Calibration requires 33..8192 pixels and a finite canvas radius above 25 mm.");
        if (Directory.Exists(outputDirectory) && Directory.EnumerateFileSystemEntries(outputDirectory).Any())
            throw new ArgumentException("Calibration output directory must be empty.");
        Directory.CreateDirectory(outputDirectory);
        byte[] image = new byte[size * size];
        List<string> measurements = ["panel,inner_mm,outer_mm,angle_start_deg,cell_um,fraction_92,linear_radiance,view_name,exposure_seconds"];
        double[] fractions = [0, .25, .5, .75, 1];
        double[] cells = [40, 80, 160];
        for (int ring = 0; ring < 3; ring++) for (int panel = 0; panel < 20; panel++)
            measurements.Add(FormattableString.Invariant($"{ring * 20 + panel},{25 + ring * 10},{35 + ring * 10},{panel * 18},{cells[ring]},{fractions[panel % 5]},,,"));
        for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
        {
            double px = (2.0 * x / (size - 1) - 1) * radius, py = (1 - 2.0 * y / (size - 1)) * radius;
            double r = Math.Sqrt(px * px + py * py); if (r < 25 || r >= 55) continue;
            int ring = (int)((r - 25) / 10);
            double angle = (Math.Atan2(py, px) * 180 / Math.PI + 360) % 360;
            int panel = Math.Min(19, (int)(angle / 18));
            int cx = (int)Math.Floor(px * 1000 / cells[ring]), cy = (int)Math.Floor(py * 1000 / cells[ring]);
            uint phase = unchecked((uint)cx * 0x9E3779B1u ^ (uint)cy * 0x85EBCA77u ^ 43523u);
            phase ^= phase >> 16; phase = unchecked(phase * 0x7FEB352Du); phase ^= phase >> 15;
            image[y * size + x] = phase / 4294967296.0 < fractions[panel % 5] ? (byte)255 : (byte)0;
        }
        RasterCodec.SavePng(Path.Combine(outputDirectory, "target.png"), size, size, image, true);
        File.WriteAllLines(Path.Combine(outputDirectory, "measurements.csv"), measurements);
        File.WriteAllText(Path.Combine(outputDirectory, "profile.json"), JsonSerializer.Serialize(new DvdPhysicalConversionSettings(), DvdPhysicalConversion.JsonOptions));
        File.WriteAllText(Path.Combine(outputDirectory, "README.txt"),
            "Experimental calibration layout; radius 58 mm, rings 25-35/35-45/45-55 mm. " +
            "Panels start at +x and run counterclockwise. Fractions repeat at four azimuths. " +
            "Target is opaque binary; use matching canvas geometry with tone processing disabled. " +
            "CSV radiance is linear and exposure-controlled; no material fit is inferred automatically. " +
            "Requested microcells below raster pitch are undersampled; increase target resolution for physical-scale measurements.");
    }
}
