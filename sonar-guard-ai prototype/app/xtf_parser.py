from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

XTF_MAGIC = b"\xf0\x5b\x28\x29"
XTF_HEADER_SIZE = 1024
PKT_PING_HEADER = 1
PKT_SNIPPET = 2
PKT_ATTITUDE = 3
PKT_RAW_SERIAL = 4
PKT_ELAPSED_TIME = 5
PKT_STRING = 7
PKT_BATHY_DATA = 53
PKT_BATHY_PROC_DATA = 54
PKT_GPS_DATA = 64
PKT_GPS_BAUD = 65
PKT_TRANSMITTER = 70
PKT_HEADER1 = 100
PKT_HEADER2 = 101
PKT_HEADER3 = 102
PKT_HEADER4 = 103
PKT_HEADER5 = 104
PKT_HEADER6 = 105
PKT_FILE_HEADER = 0


@dataclass
class GpsRecord:
    ping_index: int
    lat: float
    lon: float
    timestamp: float | None = None


@dataclass
class XtfPing:
    index: int
    port: np.ndarray | None = None
    stbd: np.ndarray | None = None


@dataclass
class XtfParseResult:
    image: np.ndarray
    gps_records: list[GpsRecord]
    ping_count: int
    origin_lat: float | None
    origin_lon: float | None


def _parse_file_header(buf: BytesIO) -> dict:
    data = buf.read(XTF_HEADER_SIZE)
    if len(data) < XTF_HEADER_SIZE:
        warnings.warn("XTF header shorter than expected; continuing with defaults")
    magic = data[:4]
    if magic != XTF_MAGIC:
        pass
    num_chans = struct.unpack_from("<H", data, 12)[0] if len(data) >= 14 else 1
    res = {"num_channels": num_chans}
    if len(data) >= 36:
        try:
            res["origin_lat"] = struct.unpack_from("<d", data, 20)[0]
            res["origin_lon"] = struct.unpack_from("<d", data, 28)[0]
        except struct.error:
            pass
    return res


def _read_packet(buf: BytesIO) -> tuple[int, int, bytes] | None:
    head = buf.read(8)
    if len(head) < 8:
        return None
    magic, pkt_type, size = struct.unpack("<2sHI", head)
    body = buf.read(max(0, size - 8)) if size > 8 else b""
    return pkt_type, size, body


def _unpack_samples(buf: bytes, offset: int, count: int, sample_bytes: int) -> np.ndarray:
    if sample_bytes == 1:
        return np.array(struct.unpack_from(f"<{count}B", buf, offset), dtype=np.uint8)
    if sample_bytes == 2:
        raw = np.array(struct.unpack_from(f"<{count}h", buf, offset), dtype=np.int16)
        norm = (raw - raw.min()).astype(np.float32)
        scale = 255.0 / max(1e-6, norm.max())
        return np.clip(norm * scale, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported sample bytes: {sample_bytes}")


def _parse_ping(ping_idx: int, body: bytes, num_samples: int | None = None) -> XtfPing:
    if len(body) < 128:
        return XtfPing(index=ping_idx)
    ping_flags = struct.unpack_from("<H", body, 8)[0] if len(body) >= 10 else 0
    nbytes_chan = 1
    if ping_flags & 0x04:
        nbytes_chan = 2
    stbd_n = struct.unpack_from("<I", body, 40)[0] if len(body) >= 44 else 0
    port_n = struct.unpack_from("<I", body, 44)[0] if len(body) >= 48 else 0
    stbd_offset = struct.unpack_from("<H", body, 56)[0] if len(body) >= 58 else 128
    port_offset = struct.unpack_from("<H", body, 58)[0] if len(body) >= 60 else 0

    if num_samples is None:
        payload = max(0, len(body) - 128)
        per_side = payload // (2 * nbytes_chan)
    else:
        per_side = num_samples

    if stbd_n and (not num_samples):
        stbd_count = min(stbd_n, 4096)
    else:
        stbd_count = max(32, min(per_side, 4096))
    if port_n and (not num_samples):
        port_count = min(port_n, 4096)
    else:
        port_count = stbd_count

    stbd = None
    port = None
    try:
        if stbd_offset and len(body) >= stbd_offset + stbd_count * nbytes_chan:
            stbd = _unpack_samples(body, stbd_offset, stbd_count, nbytes_chan)
        if port_offset and len(body) >= port_offset + port_count * nbytes_chan:
            port = _unpack_samples(body, port_offset, port_count, nbytes_chan)
    except (struct.error, ValueError):
        stbd = None
        port = None

    if stbd is None and port is None:
        payload = max(0, len(body) - 128)
        per_side = payload // 2
        per_side = max(32, min(per_side, 4096))
        fmt = f"<{per_side}B"
        if len(body) >= 128 + per_side:
            try:
                stbd = np.array(struct.unpack_from(fmt, body, 128), dtype=np.uint8)
            except struct.error:
                stbd = None
        if len(body) >= 128 + 2 * per_side:
            try:
                port = np.array(struct.unpack_from(fmt, body, 128 + per_side), dtype=np.uint8)
            except struct.error:
                port = None
    return XtfPing(index=ping_idx, port=port, stbd=stbd)


def _parse_gps(body: bytes) -> GpsRecord | None:
    if len(body) < 40:
        return None
    try:
        ping_idx = struct.unpack_from("<I", body, 0)[0]
        lat = struct.unpack_from("<d", body, 8)[0]
        lon = struct.unpack_from("<d", body, 16)[0]
        ts = struct.unpack_from("<d", body, 24)[0] if len(body) >= 32 else None
        if abs(lat) > 90 or abs(lon) > 180:
            return None
        return GpsRecord(ping_index=ping_idx, lat=lat, lon=lon, timestamp=ts)
    except struct.error:
        return None


def parse_xtf(data: bytes | str | Path) -> XtfParseResult:
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    buf = BytesIO(data)
    hdr = _parse_file_header(buf)
    num_chans = max(1, min(hdr.get("num_channels", 1), 2))
    pings: list[XtfPing] = []
    gps: list[GpsRecord] = []
    ping_counter = 0
    while True:
        pkt = _read_packet(buf)
        if pkt is None:
            break
        pkt_type, _size, body = pkt
        if pkt_type == PKT_PING_HEADER or pkt_type == PKT_SNIPPET:
            ping = _parse_ping(ping_counter, body)
            if ping.stbd is not None or ping.port is not None:
                pings.append(ping)
                ping_counter += 1
        elif pkt_type == PKT_GPS_DATA:
            rec = _parse_gps(body)
            if rec is not None:
                gps.append(rec)
    if not pings:
        raise ValueError("XTF file contained no readable pings")
    side = pings[0].stbd if pings[0].stbd is not None else pings[0].port
    width = side.shape[0] if side is not None else 256
    waterfall = np.zeros((len(pings), width * num_chans), dtype=np.uint8)
    for i, ping in enumerate(pings):
        if ping.stbd is not None:
            waterfall[i, :width] = ping.stbd[:width]
        if ping.port is not None and num_chans >= 2:
            w = min(ping.port.shape[0], width)
            waterfall[i, width : width + w] = ping.port[:w]
    origin_lat = gps[0].lat if gps else None
    origin_lon = gps[0].lon if gps else None
    return XtfParseResult(
        image=waterfall,
        gps_records=[
            {"ping_index": r.ping_index, "lat": r.lat, "lon": r.lon, "timestamp": r.timestamp}
            for r in gps
        ],
        ping_count=len(pings),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def _build_synthetic_xtf(num_pings: int, samples_per_side: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    header = bytearray(XTF_HEADER_SIZE)
    header[:4] = XTF_MAGIC
    struct.pack_into("<H", header, 12, 2)
    out = bytearray(bytes(header))
    base_lat = 13.0480
    base_lon = 80.4120
    for i in range(num_pings):
        stbd = np.clip(rng.normal(60, 18, samples_per_side), 0, 255).astype(np.uint8)
        port = np.clip(rng.normal(55, 16, samples_per_side), 0, 255).astype(np.uint8)
        obj_x = 120 + (i * 37) % (samples_per_side - 80)
        stbd[obj_x : obj_x + 24] = np.clip(stbd[obj_x : obj_x + 24].astype(np.int16) + 150, 0, 255).astype(np.uint8)
        body = bytearray(128)
        body.extend(bytes(stbd))
        body.extend(bytes(port))
        pkt = bytearray()
        pkt.extend(struct.pack("<2sHI", b"XT", PKT_PING_HEADER, 8 + len(body)))
        pkt.extend(bytes(body))
        out.extend(pkt)
        lat = base_lat + (i * 0.00002)
        lon = base_lon + (i * 0.00001)
        gps_body = bytearray(48)
        struct.pack_into("<I", gps_body, 0, i)
        struct.pack_into("<d", gps_body, 8, lat)
        struct.pack_into("<d", gps_body, 16, lon)
        struct.pack_into("<d", gps_body, 24, 1_700_000_000.0 + i * 0.1)
        gps_pkt = bytearray()
        gps_pkt.extend(struct.pack("<2sHI", b"XT", PKT_GPS_DATA, 8 + len(gps_body)))
        gps_pkt.extend(bytes(gps_body))
        out.extend(gps_pkt)
    return bytes(out)


def px_to_gps(ping_idx: int, across_px: int, gps_records: list[dict], frame_shape: tuple[int, int], mpp_along: float, mpp_across: float) -> tuple[float, float]:
    h, w = frame_shape
    if not gps_records:
        return (0.0, 0.0)
    recs = sorted(gps_records, key=lambda r: r["ping_index"])
    if ping_idx <= recs[0]["ping_index"]:
        lat, lon = recs[0]["lat"], recs[0]["lon"]
    elif ping_idx >= recs[-1]["ping_index"]:
        lat, lon = recs[-1]["lat"], recs[-1]["lon"]
    else:
        lo, hi = 0, len(recs) - 1
        for idx in range(len(recs) - 1):
            if recs[idx]["ping_index"] <= ping_idx <= recs[idx + 1]["ping_index"]:
                lo, hi = idx, idx + 1
                break
        a, b = recs[lo], recs[hi]
        span = max(1, b["ping_index"] - a["ping_index"])
        t = (ping_idx - a["ping_index"]) / span
        lat = a["lat"] + (b["lat"] - a["lat"]) * t
        lon = a["lon"] + (b["lon"] - a["lon"]) * t
    half_w = w / 2.0
    across_m = (across_px - half_w) * mpp_across
    dlat = 0.0
    dlon = across_m / (111_320.0 * np.cos(np.deg2rad(lat)))
    return lat + dlat, lon + dlon


def self_test() -> None:
    data = _build_synthetic_xtf(num_pings=4, samples_per_side=128, seed=42)
    result = parse_xtf(data)
    assert result.image.shape[0] == 4, f"expected 4 pings, got {result.image.shape[0]}"
    assert len(result.gps_records) == 4, f"expected 4 gps records, got {len(result.gps_records)}"
    print(f"[xtf self_test] OK: image shape={result.image.shape}, gps_records={len(result.gps_records)}")


def write_synthetic_xtf(out_path: str | Path, num_pings: int = 128, samples_per_side: int = 512, seed: int = 42) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_build_synthetic_xtf(num_pings, samples_per_side, seed))
    return out


CHENNAI_SURVEY_PARAMS = {
    "kmg_unit": "Klein Marine Systems 3900-Side (400 kHz)",
    "alongtrack_m_per_ping": 0.05,
    "acrosstrack_m_per_sample": 0.04,
    "ping_rate_hz": 10,
    "vessel_speed_knots": 2.5,
    "frequency_khz": 400,
    "max_across_range_m": 50,
    "along_slant_heave_m": 0.02,
    "sample_rate_khz": 250,
    "origin_lat": 13.0480,
    "origin_lon": 80.4120,
}


def _write_pkt(out: bytearray, pkt_type: int, body: bytes) -> None:
    out.extend(struct.pack("<2sHI", b"XT", pkt_type, 8 + len(body)))
    out.extend(body)


def _build_realistic_survey_xtf(num_pings: int = 512, samples_per_side: int = 1024,
                                seed: int = 2026, realistic: bool = True) -> bytes:
    rng = np.random.default_rng(seed)
    params = CHENNAI_SURVEY_PARAMS
    along_m_per_ping = params["alongtrack_m_per_ping"]
    across_m_per_samp = params["acrosstrack_m_per_sample"]

    header = bytearray(XTF_HEADER_SIZE)
    header[:4] = XTF_MAGIC
    struct.pack_into("<H", header, 12, 2)
    struct.pack_into("<d", header, 20, params["origin_lat"])
    struct.pack_into("<d", header, 28, params["origin_lon"])
    note = params["kmg_unit"].encode("ascii")
    header[40 : 40 + min(len(note), XTF_HEADER_SIZE - 40)] = note
    out = bytearray(bytes(header))

    chaninfo_body = bytearray(256)
    struct.pack_into("<I", chaninfo_body, 0, 400)
    struct.pack_into("<f", chaninfo_body, 4, params["max_across_range_m"])
    struct.pack_into("<H", chaninfo_body, 8, samples_per_side)
    struct.pack_into("<f", chaninfo_body, 10, params["sample_rate_khz"])
    _write_pkt(out, PKT_HEADER1, bytes(chaninfo_body))

    base_lat = params["origin_lat"]
    base_lon = params["origin_lon"]
    ping_interval_s = 1.0 / params["ping_rate_hz"]
    speed_m_s = params["vessel_speed_knots"] * 0.5144
    heading_deg = 45.0
    hdg = np.deg2rad(heading_deg)

    t0 = 1_700_000_000.0

    targets: list[dict] = []
    if realistic:
        n_targets = rng.integers(8, 14) if num_pings > 120 else rng.integers(1, 3)
        ping_lo = 5 if num_pings <= 40 else 20 if num_pings <= 100 else 40
        ping_hi = num_pings - ping_lo if num_pings - ping_lo > ping_lo else num_pings - 1
        samp_lo = 8 if samples_per_side < 240 else 60 if samples_per_side < 360 else 120
        samp_hi = samples_per_side - samp_lo if samples_per_side - samp_lo > samp_lo else samples_per_side - 1
        scale = max(1.0, samples_per_side / 512.0)
        along_min = max(8, int(14 * scale))
        along_max = max(24, int(60 * scale))
        across_min = max(14, int(22 * scale))
        across_max = max(48, int(120 * scale))
        for _ in range(n_targets):
            targets.append({
                "ping_i": int(rng.uniform(ping_lo, ping_hi)),
                "side": rng.choice(["port", "stbd"]),
                "across_samp": int(rng.uniform(samp_lo, samp_hi)),
                "along_size_pings": int(rng.uniform(along_min, along_max)),
                "across_size_samps": int(rng.uniform(across_min, across_max)),
                "strength": rng.uniform(0.7, 1.0),
                "class_idx": int(rng.integers(0, 4)),
                "has_shadow": rng.random() > 0.15,
            })

    seabed_bands = [
        (rng.integers(220, 260), rng.integers(340, 420), rng.integers(10, 22)),
    ]
    stripe_period = rng.integers(30, 80)

    for i in range(num_pings):
        stbd_raw = np.clip(rng.normal(62, 14, samples_per_side), 0, 255)
        port_raw = np.clip(rng.normal(58, 13, samples_per_side), 0, 255)

        slow = rng.normal() * 0.6
        stbd_raw = np.clip(stbd_raw + slow * 10, 0, 255)
        port_raw = np.clip(port_raw + slow * 9, 0, 255)

        stripe_wave = 4 * np.sin(2 * np.pi * (i % stripe_period) / stripe_period)
        stbd_raw = np.clip(stbd_raw + stripe_wave, 0, 255)
        port_raw = np.clip(port_raw + stripe_wave, 0, 255)

        tvg = np.linspace(0.7, 1.3, samples_per_side)
        stbd_raw = np.clip(stbd_raw * tvg, 0, 255)
        port_raw = np.clip(port_raw * tvg[::-1], 0, 255)

        if realistic:
            for lo, hi, amp in seabed_bands:
                envelope = np.exp(-0.5 * ((np.arange(samples_per_side) - (lo + hi) / 2) / ((hi - lo) / 4)) ** 2)
                stbd_raw = np.clip(stbd_raw + amp * envelope * 40, 0, 255)
                port_raw = np.clip(port_raw + amp * envelope[::-1] * 38, 0, 255)

        stbd = stbd_raw.astype(np.uint8)
        port = port_raw.astype(np.uint8)

        for t in targets:
            side = t["side"]
            ping_span = (max(0, i - (t["ping_i"] - t["along_size_pings"] // 2)),
                         min(num_pings - 1, i - (t["ping_i"] - t["along_size_pings"] // 2) + t["along_size_pings"]))
            rel = (i - (t["ping_i"] - t["along_size_pings"] // 2)) / max(1, t["along_size_pings"])
            if 0.0 <= rel <= 1.0:
                window = np.sin(np.pi * rel) ** 2
                a0 = t["across_samp"] - t["across_size_samps"] // 2
                a1 = t["across_samp"] + t["across_size_samps"] // 2
                a0 = max(0, a0)
                a1 = min(samples_per_side - 1, a1)
                add = int(180 * t["strength"] * window)
                if side == "stbd":
                    stbd[a0:a1] = np.clip(stbd[a0:a1].astype(np.int16) + add, 0, 255).astype(np.uint8)
                    if t["has_shadow"] and a1 + t["across_size_samps"] < samples_per_side:
                        sh0, sh1 = a1, min(samples_per_side, a1 + t["across_size_samps"])
                        stbd[sh0:sh1] = np.clip(stbd[sh0:sh1].astype(np.int16) - int(90 * window), 0, 255).astype(np.uint8)
                else:
                    port[a0:a1] = np.clip(port[a0:a1].astype(np.int16) + add, 0, 255).astype(np.uint8)
                    if t["has_shadow"] and a1 + t["across_size_samps"] < samples_per_side:
                        sh0, sh1 = a1, min(samples_per_side, a1 + t["across_size_samps"])
                        port[sh0:sh1] = np.clip(port[sh0:sh1].astype(np.int16) - int(90 * window), 0, 255).astype(np.uint8)

        ping_body = bytearray(128)
        struct.pack_into("<I", ping_body, 0, i)
        struct.pack_into("<f", ping_body, 4, ping_interval_s * i)
        struct.pack_into("<I", ping_body, 40, samples_per_side)
        struct.pack_into("<I", ping_body, 44, samples_per_side)
        struct.pack_into("<H", ping_body, 56, 128)
        struct.pack_into("<H", ping_body, 58, 128 + samples_per_side)
        ping_body.extend(bytes(stbd))
        ping_body.extend(bytes(port))
        _write_pkt(out, PKT_PING_HEADER, bytes(ping_body))

        dist_m = i * along_m_per_ping * speed_m_s / (speed_m_s)
        dlat_m = np.cos(hdg) * dist_m
        dlon_m = np.sin(hdg) * dist_m
        lat = base_lat + dlat_m / 111_320.0
        lon = base_lon + dlon_m / (111_320.0 * np.cos(np.deg2rad(base_lat)))
        if realistic and i > 0:
            lat += rng.normal(0, 0.000002)
            lon += rng.normal(0, 0.000003)

        gps_body = bytearray(48)
        struct.pack_into("<I", gps_body, 0, i)
        struct.pack_into("<d", gps_body, 8, lat)
        struct.pack_into("<d", gps_body, 16, lon)
        struct.pack_into("<d", gps_body, 24, t0 + i * ping_interval_s)
        struct.pack_into("<f", gps_body, 32, 1.4)
        struct.pack_into("<f", gps_body, 36, heading_deg)
        _write_pkt(out, PKT_GPS_DATA, bytes(gps_body))

        if i % 16 == 0:
            att_body = bytearray(48)
            struct.pack_into("<I", att_body, 0, i)
            struct.pack_into("<f", att_body, 4, float(rng.normal(0, 0.6)))
            struct.pack_into("<f", att_body, 8, float(rng.normal(0, 0.4)))
            struct.pack_into("<f", att_body, 12, float(rng.normal(0, 0.2)))
            struct.pack_into("<d", att_body, 16, t0 + i * ping_interval_s)
            _write_pkt(out, PKT_ATTITUDE, bytes(att_body))

    summary_body = (
        f"SONARGUARD_SIM survey=CHENNAI_BAY pings={num_pings} "
        f"samples={samples_per_side} freq_khz={params['frequency_khz']} "
        f"mpp_along={along_m_per_ping:.4f} mpp_across={across_m_per_samp:.4f} "
        f"targets={len(targets)}"
    ).encode("ascii")
    str_body = bytearray(32)
    str_body.extend(summary_body)
    _write_pkt(out, PKT_STRING, bytes(str_body))

    return bytes(out)


def write_realistic_survey_xtf(out_path: str | Path, num_pings: int = 512, samples_per_side: int = 1024,
                               seed: int = 2026, realistic: bool = True) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_build_realistic_survey_xtf(num_pings, samples_per_side, seed, realistic))
    return out


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic XTF log generator for SonarGuard AI")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output .xtf path (omit for --self-test)")
    parser.add_argument("--pings", type=int, default=128, help="Number of sonar pings (default: 128)")
    parser.add_argument("--samples", type=int, default=512, help="Samples per side per ping (default: 512)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible synthetic data")
    parser.add_argument("--realistic", action="store_true",
                        help="Generate a survey-calibrated realistic file (GPS track, TVG, seabed textures, 6-12 targets, attitudes)")
    parser.add_argument("--self-test", action="store_true", help="Run parser self-tests and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        data = _build_realistic_survey_xtf(num_pings=16, samples_per_side=256, seed=7, realistic=True)
        r = parse_xtf(data)
        assert r.image.shape == (16, 512), r.image.shape
        assert len(r.gps_records) == 16
        assert r.origin_lat is not None and r.origin_lon is not None
        print(f"[xtf self_test realistic] OK: shape={r.image.shape}, gps={len(r.gps_records)}, "
              f"origin=({r.origin_lat:.4f}, {r.origin_lon:.4f})")
        return

    if args.output is None:
        parser.error("output is required unless --self-test is used")

    if args.realistic:
        path = write_realistic_survey_xtf(args.output, num_pings=args.pings, samples_per_side=args.samples,
                                          seed=args.seed, realistic=True)
    else:
        path = write_synthetic_xtf(args.output, num_pings=args.pings, samples_per_side=args.samples, seed=args.seed)
    print(f"Wrote {'realistic ' if args.realistic else ''}XTF with {args.pings} pings x {args.samples} samples/side -> {path}")


if __name__ == "__main__":
    _cli()
