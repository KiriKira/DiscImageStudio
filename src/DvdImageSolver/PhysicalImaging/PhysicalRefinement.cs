using System.Buffers;

namespace DvdImageSolver.PhysicalImaging;

internal static class PhysicalRefinement
{
    // Coordinate descent in non-overlapping tiles. Each accepted flip updates its
    // complete PSF footprint. Halo errors include neighboring tiles, avoiding seams.
    public static double[] Refine(byte[] bits, float[] tone, double[] lo, double[] hi, bool[] mask, int size, double cell,
        float[][] maps, int m, DvdPhysicalConversionSettings settings, CancellationToken token,
        Action<string, double>? progress)
    {
        const int tile = 128;
        List<double> improvements = [];
        for (int pass = 0; pass < settings.Passes; pass++)
        {
            double improvement = 0, baseline = 0;
            for (int top = 0; top < size; top += tile) for (int left = 0; left < size; left += tile)
            {
                token.ThrowIfCancellationRequested();
                progress?.Invoke($"Physical optimization {pass + 1}/{settings.Passes}",
                    .5 + .49 * (pass + (top * (double)size + left) / (size * (double)size)) / settings.Passes);
                int right = Math.Min(size, left + tile), bottom = Math.Min(size, top + tile);
                // Fine cells can require large kernels; PSFs are always in mm.
                double[] scales = [.02, .06, .18], scaleWeights = [1, .5, .25];
                List<(double[] Error, double[] Kernel, int Radius, double Weight, int X, int Y, int W, int H, int View)> fields = [];
                for (int v = 0; v < settings.Views.Length; v++) for (int s = 0; s < scales.Length; s++)
                {
                    double sigma = scales[s] / cell; int r = Math.Max(1, (int)Math.Ceiling(3 * sigma));
                    double[] k = new double[2 * r + 1]; double sum = 0;
                    for (int j = -r; j <= r; j++) { k[j + r] = Math.Exp(-.5 * j * j / (sigma * sigma)); sum += k[j + r]; }
                    for (int j = 0; j < k.Length; j++) k[j] /= sum;
                    int x0 = Math.Max(0, left - 2 * r), y0 = Math.Max(0, top - 2 * r);
                    int w = Math.Min(size, right + 2 * r) - x0, h = Math.Min(size, bottom + 2 * r) - y0;
                    double[] raw = ArrayPool<double>.Shared.Rent(w * h), horizontal = ArrayPool<double>.Shared.Rent(w * h), filtered = ArrayPool<double>.Shared.Rent(w * h);
                    Array.Clear(raw, 0, w * h); Array.Clear(horizontal, 0, w * h); Array.Clear(filtered, 0, w * h);
                    for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++)
                    {
                        int gx = x0 + xx, gy = y0 + yy, i = gy * size + gx;
                        if (!mask[i]) continue;
                        double d = DvdPhysicalConversion.Sample(maps[2 * v + 1], m, gx, gy, size) - DvdPhysicalConversion.Sample(maps[2 * v], m, gx, gy, size);
                        raw[yy * w + xx] = DvdPhysicalConversion.Sample(maps[2 * v], m, gx, gy, size)
                            + d * bits[i] / 255.0 - lo[v] - (hi[v] - lo[v]) * tone[i];
                    }
                    for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++)
                        for (int j = -r; j <= r; j++) if (xx + j >= 0 && xx + j < w)
                            horizontal[yy * w + xx] += k[j + r] * raw[yy * w + xx + j];
                    for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++)
                        for (int j = -r; j <= r; j++) if (yy + j >= 0 && yy + j < h)
                            filtered[yy * w + xx] += k[j + r] * horizontal[(yy + j) * w + xx];
                    for (int gy = top; gy < bottom; gy++) for (int gx = left; gx < right; gx++)
                        if (mask[gy * size + gx]) baseline += scaleWeights[s] * settings.Views[v].Weight * Math.Pow(filtered[(gy - y0) * w + gx - x0], 2);
                    ArrayPool<double>.Shared.Return(raw);
                    ArrayPool<double>.Shared.Return(horizontal);
                    fields.Add((filtered, k, r, scaleWeights[s] * settings.Views[v].Weight,
                        x0, y0, w, h, v));
                }
                // Sparse deterministic sweeps bound cost; successive passes change phase.
                for (int yy = top; yy < bottom; yy++) for (int xx = left + (yy + pass) % 4; xx < right; xx += 4)
                {
                    int i = yy * size + xx; if (!mask[i]) continue;
                    if (!TryChange(xx, yy, -1, -1))
                    {
                        if (xx + 1 < right && mask[i + 1] && bits[i + 1] != bits[i]) TryChange(xx, yy, xx + 1, yy);
                        else if (yy + 1 < bottom && mask[i + size] && bits[i + size] != bits[i]) TryChange(xx, yy, xx, yy + 1);
                    }
                    bool TryChange(int ax, int ay, int bx, int by)
                    {
                        int ia = ay * size + ax, signA = bits[ia] == 0 ? 1 : -1;
                        bool pair = bx >= 0; int ib = pair ? by * size + bx : ia;
                        int signB = pair ? (bits[ib] == 0 ? 1 : -1) : 0;
                        double change = 0;
                        foreach (var f in fields)
                        {
                            double da = signA * (DvdPhysicalConversion.Sample(maps[f.View * 2 + 1], m, ax, ay, size) - DvdPhysicalConversion.Sample(maps[f.View * 2], m, ax, ay, size));
                            double db = pair ? signB * (DvdPhysicalConversion.Sample(maps[f.View * 2 + 1], m, bx, by, size) - DvdPhysicalConversion.Sample(maps[f.View * 2], m, bx, by, size)) : 0;
                            int minX = Math.Max(0, Math.Min(ax, pair ? bx : ax) - f.Radius), maxX = Math.Min(size - 1, Math.Max(ax, pair ? bx : ax) + f.Radius);
                            int minY = Math.Max(0, Math.Min(ay, pair ? by : ay) - f.Radius), maxY = Math.Min(size - 1, Math.Max(ay, pair ? by : ay) + f.Radius);
                            for (int gy = minY; gy <= maxY; gy++) for (int gx = minX; gx <= maxX; gx++)
                            {
                                if (!mask[gy * size + gx]) continue;
                                double delta = Kernel(gx - ax, gy - ay) * da + (pair ? Kernel(gx - bx, gy - by) * db : 0);
                                change += f.Weight * (2 * f.Error[(gy - f.Y) * f.W + gx - f.X] * delta + delta * delta);
                            }
                            double Kernel(int dx, int dy) => Math.Abs(dx) <= f.Radius && Math.Abs(dy) <= f.Radius ? f.Kernel[dx + f.Radius] * f.Kernel[dy + f.Radius] : 0;
                        }
                        if (change >= -1e-24) return false;
                        bits[ia] = (byte)(255 - bits[ia]); if (pair) bits[ib] = (byte)(255 - bits[ib]); improvement -= change;
                        foreach (var f in fields)
                        {
                            Update(ax, ay, signA); if (pair) Update(bx, by, signB);
                            void Update(int px, int py, int sign)
                            {
                                double d = sign * (DvdPhysicalConversion.Sample(maps[f.View * 2 + 1], m, px, py, size) - DvdPhysicalConversion.Sample(maps[f.View * 2], m, px, py, size));
                                for (int ky = -f.Radius; ky <= f.Radius; ky++) for (int kx = -f.Radius; kx <= f.Radius; kx++)
                                {
                                    int gx = px + kx, gy = py + ky;
                                    if (gx < f.X || gy < f.Y || gx >= f.X + f.W || gy >= f.Y + f.H) continue;
                                    f.Error[(gy - f.Y) * f.W + gx - f.X] += d * f.Kernel[kx + f.Radius] * f.Kernel[ky + f.Radius];
                                }
                            }
                        }
                        return true;
                    }
                }
                foreach (var field in fields) ArrayPool<double>.Shared.Return(field.Error);
            }
            improvements.Add(improvement);
            if (improvement <= Math.Max(1e-24, baseline * .0001)) break;
        }
        return improvements.ToArray();
    }
}
