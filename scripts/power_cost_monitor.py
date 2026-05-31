#!/usr/bin/env python3
"""Measure model-run energy usage and estimate San Diego electricity cost.

This wrapper launches a training/inference command, samples available Ubuntu
power sensors while it runs, integrates energy into kWh, and appends a per-run
cost record. It is intentionally dependency-free.

Default rate model:
  SDG&E residential Schedule TOU-DR1 bundled total electric rates, effective
  2026-04-01. TOU-DR1 is SDG&E's standard residential time-of-use schedule for
  a typical household. The script excludes fixed base service charges because
  model runs add marginal kWh, not monthly account days.

Usage:
    python scripts/power_cost_monitor.py --label grpo-smoke -- \
      python scripts/train_grpo_full.py --train-limit 16 --eval-limit 16 \
        --max-steps 5 --num-generations 4 --max-completion-length 1024

Notes:
  - NVIDIA GPU power uses nvidia-smi.
  - CPU package energy uses Linux powercap/RAPL counters when readable.
  - Software sensors do not capture the whole wall outlet. Use --extra-watts
    for motherboard/fans/PSU losses/monitor overhead, or --fixed-watts if you
    measured the whole PC with a wall meter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


DEFAULT_JSONL_LOG = Path("results/power_usage_runs.jsonl")
DEFAULT_CSV_LOG = Path("results/power_usage_runs.csv")
DEFAULT_TIMEZONE = "America/Los_Angeles"

SDGE_BASELINE_CREDIT_USD_PER_KWH = 0.10892
SDGE_TOUDR1_TOTAL_RATES: dict[str, dict[str, float]] = {
    # Total Electric Rate before the baseline adjustment credit.
    "summer": {
        "super_off_peak": 0.38773,
        "off_peak": 0.47505,
        "on_peak": 0.69572,
    },
    "winter": {
        "super_off_peak": 0.44880,
        "off_peak": 0.53956,
        "on_peak": 0.62127,
    },
}


@dataclass(frozen=True)
class RateResult:
    usd_per_kwh: float
    mode: str
    period: str
    season: str
    tier: str


@dataclass
class SensorSnapshot:
    timestamp_monotonic: float
    timestamp_wall: datetime
    gpu_watts: float | None
    rapl_joules_by_domain: dict[str, float]


@dataclass
class IntervalEnergy:
    elapsed_s: float
    gpu_j: float
    cpu_j: float
    extra_j: float
    fixed_j: float
    fallback_j: float
    rate: RateResult

    @property
    def total_j(self) -> float:
        return self.gpu_j + self.cpu_j + self.extra_j + self.fixed_j + self.fallback_j

    @property
    def total_kwh(self) -> float:
        return joules_to_kwh(self.total_j)

    @property
    def cost_usd(self) -> float:
        return self.total_kwh * self.rate.usd_per_kwh


class NvidiaSmiReader:
    """Reads instantaneous NVIDIA GPU board power in watts."""

    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.available = self.executable is not None
        self.last_error: str | None = None

    def read_total_watts(self) -> float | None:
        if not self.executable:
            return None
        try:
            proc = subprocess.run(
                [
                    self.executable,
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.available = False
            self.last_error = str(exc)
            return None

        if proc.returncode != 0:
            self.last_error = proc.stderr.strip() or proc.stdout.strip()
            return None

        watts: list[float] = []
        for line in proc.stdout.splitlines():
            text = line.strip()
            if not text or text.upper() in {"N/A", "[N/A]"}:
                continue
            try:
                watts.append(float(text))
            except ValueError:
                continue

        if not watts:
            self.last_error = "nvidia-smi returned no numeric power.draw values"
            return None
        self.available = True
        self.last_error = None
        return sum(watts)


class RaplReader:
    """Reads Linux powercap/RAPL package energy counters in joules."""

    def __init__(self, root: Path = Path("/sys/class/powercap")) -> None:
        self.root = root
        self.domains = self._discover_domains(root)
        self.available = bool(self.domains)
        self.last_error: str | None = None

    @staticmethod
    def _discover_domains(root: Path) -> list[tuple[str, Path, float | None]]:
        domains: list[tuple[str, Path, float | None]] = []
        if not root.exists():
            return domains

        try:
            candidates = sorted(root.iterdir())
        except OSError:
            return domains

        for domain in candidates:
            if not domain.is_dir():
                continue
            energy_path = domain / "energy_uj"
            if not energy_path.exists():
                continue

            name = read_text(domain / "name") or domain.name
            if not (
                domain.name.startswith("intel-rapl:")
                or domain.name.startswith("amd-rapl:")
                or "package" in name.lower()
            ):
                continue

            max_range_text = read_text(domain / "max_energy_range_uj")
            max_range_j = None
            if max_range_text:
                try:
                    max_range_j = float(max_range_text) / 1_000_000.0
                except ValueError:
                    max_range_j = None

            label = f"{domain.name}:{name}"
            domains.append((label, energy_path, max_range_j))

        return domains

    def read_joules_by_domain(self) -> dict[str, float]:
        values: dict[str, float] = {}
        errors: list[str] = []
        for label, path, _max_range_j in self.domains:
            try:
                values[label] = float(path.read_text().strip()) / 1_000_000.0
            except (OSError, ValueError) as exc:
                errors.append(f"{label}: {exc}")
        self.last_error = "; ".join(errors) if errors else None
        return values

    def delta_joules(self, previous: dict[str, float], current: dict[str, float]) -> float:
        total = 0.0
        max_ranges = {label: max_range_j for label, _path, max_range_j in self.domains}
        for label, prev_value in previous.items():
            if label not in current:
                continue
            cur_value = current[label]
            delta = cur_value - prev_value
            if delta < 0:
                max_range_j = max_ranges.get(label)
                if max_range_j:
                    delta = (max_range_j - prev_value) + cur_value
            if delta > 0 and math.isfinite(delta):
                total += delta
        return total


class PowerMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tz = ZoneInfo(args.timezone)
        self.gpu_reader = NvidiaSmiReader() if args.power_source == "sensors" and args.gpu else None
        self.rapl_reader = RaplReader() if args.power_source == "sensors" and args.cpu else None
        self.used_gpu_sensor = False
        self.used_cpu_sensor = False
        self.used_fallback = False

    def snapshot(self) -> SensorSnapshot:
        now_mono = time.monotonic()
        now_wall = datetime.now(self.tz)
        gpu_watts = self.gpu_reader.read_total_watts() if self.gpu_reader else None
        rapl = self.rapl_reader.read_joules_by_domain() if self.rapl_reader else {}
        if gpu_watts is not None:
            self.used_gpu_sensor = True
        if rapl:
            self.used_cpu_sensor = True
        return SensorSnapshot(now_mono, now_wall, gpu_watts, rapl)

    def interval_energy(self, prev: SensorSnapshot, cur: SensorSnapshot) -> IntervalEnergy:
        dt = max(0.0, cur.timestamp_monotonic - prev.timestamp_monotonic)
        midpoint = prev.timestamp_wall + (cur.timestamp_wall - prev.timestamp_wall) / 2
        rate = rate_for_time(midpoint, self.args)

        if self.args.power_source == "fixed":
            fixed_j = self.args.fixed_watts * dt
            return IntervalEnergy(dt, 0.0, 0.0, 0.0, fixed_j, 0.0, rate)

        gpu_j = 0.0
        if prev.gpu_watts is not None and cur.gpu_watts is not None:
            gpu_j = ((prev.gpu_watts + cur.gpu_watts) / 2.0) * dt
        elif cur.gpu_watts is not None:
            gpu_j = cur.gpu_watts * dt
        elif prev.gpu_watts is not None:
            gpu_j = prev.gpu_watts * dt

        cpu_j = 0.0
        if self.rapl_reader and prev.rapl_joules_by_domain and cur.rapl_joules_by_domain:
            cpu_j = self.rapl_reader.delta_joules(prev.rapl_joules_by_domain, cur.rapl_joules_by_domain)

        extra_j = self.args.extra_watts * dt
        fallback_j = 0.0
        if self.args.fallback_watts > 0 and gpu_j == 0.0 and cpu_j == 0.0:
            fallback_j = self.args.fallback_watts * dt
            self.used_fallback = True

        return IntervalEnergy(dt, gpu_j, cpu_j, extra_j, 0.0, fallback_j, rate)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def joules_to_kwh(joules: float) -> float:
    return joules / 3_600_000.0


def is_summer(day: date) -> bool:
    return date(day.year, 6, 1) <= day <= date(day.year, 10, 31)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cur = date(year, month, 1)
    offset = (weekday - cur.weekday()) % 7
    return cur + timedelta(days=offset + (n - 1) * 7)


def last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cur = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cur = date(year, month + 1, 1) - timedelta(days=1)
    offset = (cur.weekday() - weekday) % 7
    return cur - timedelta(days=offset)


def sdge_holidays(year: int) -> set[date]:
    return {
        date(year, 1, 1),
        nth_weekday(year, 2, 0, 3),  # Presidents' Day.
        last_weekday(year, 5, 0),  # Memorial Day.
        date(year, 7, 4),
        nth_weekday(year, 9, 0, 1),  # Labor Day.
        date(year, 11, 11),
        nth_weekday(year, 11, 3, 4),  # Thanksgiving Day.
        date(year, 12, 25),
    }


def tou_dr1_period(moment: datetime) -> str:
    local_time = moment.timetz().replace(tzinfo=None)
    weekend_or_holiday = moment.weekday() >= 5 or moment.date() in sdge_holidays(moment.year)

    if weekend_or_holiday:
        if dt_time(0, 0) <= local_time < dt_time(14, 0):
            return "super_off_peak"
        if dt_time(16, 0) <= local_time < dt_time(21, 0):
            return "on_peak"
        return "off_peak"

    if dt_time(0, 0) <= local_time < dt_time(6, 0):
        return "super_off_peak"
    if dt_time(10, 0) <= local_time < dt_time(14, 0):
        return "super_off_peak"
    if dt_time(16, 0) <= local_time < dt_time(21, 0):
        return "on_peak"
    return "off_peak"


def rate_for_time(moment: datetime, args: argparse.Namespace) -> RateResult:
    if args.rate_usd_per_kwh is not None:
        return RateResult(args.rate_usd_per_kwh, "flat", "flat", "all", "custom")

    season = "summer" if is_summer(moment.date()) else "winter"
    period = tou_dr1_period(moment)
    rate = SDGE_TOUDR1_TOTAL_RATES[season][period]
    if args.sdge_tier == "tier1":
        rate -= SDGE_BASELINE_CREDIT_USD_PER_KWH
    return RateResult(rate, "sdge-tou-dr1-bundled", period, season, args.sdge_tier)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def append_csv(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "label",
        "start_time",
        "end_time",
        "duration_s",
        "exit_code",
        "total_kwh",
        "cost_usd",
        "avg_total_watts",
        "gpu_kwh",
        "cpu_kwh",
        "extra_kwh",
        "fixed_kwh",
        "fallback_kwh",
        "rate_mode",
        "sdge_tier",
        "command",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fieldnames})


def open_sample_writer(path_text: str | None) -> tuple[csv.DictWriter, object] | tuple[None, None]:
    if not path_text:
        return None, None
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "timestamp",
        "elapsed_s",
        "gpu_watts_avg",
        "cpu_watts_avg",
        "extra_watts",
        "fixed_watts",
        "fallback_watts",
        "total_watts_avg",
        "rate_usd_per_kwh",
        "rate_period",
        "rate_season",
        "interval_kwh",
        "interval_cost_usd",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def write_sample(writer: csv.DictWriter | None, start: SensorSnapshot, interval: IntervalEnergy) -> None:
    if not writer or interval.elapsed_s <= 0:
        return
    total_watts = interval.total_j / interval.elapsed_s
    writer.writerow(
        {
            "timestamp": start.timestamp_wall.isoformat(),
            "elapsed_s": f"{interval.elapsed_s:.6f}",
            "gpu_watts_avg": f"{interval.gpu_j / interval.elapsed_s:.6f}",
            "cpu_watts_avg": f"{interval.cpu_j / interval.elapsed_s:.6f}",
            "extra_watts": f"{interval.extra_j / interval.elapsed_s:.6f}",
            "fixed_watts": f"{interval.fixed_j / interval.elapsed_s:.6f}",
            "fallback_watts": f"{interval.fallback_j / interval.elapsed_s:.6f}",
            "total_watts_avg": f"{total_watts:.6f}",
            "rate_usd_per_kwh": f"{interval.rate.usd_per_kwh:.6f}",
            "rate_period": interval.rate.period,
            "rate_season": interval.rate.season,
            "interval_kwh": f"{interval.total_kwh:.9f}",
            "interval_cost_usd": f"{interval.cost_usd:.6f}",
        }
    )


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if Path("/proc").exists():
        return Path(f"/proc/{pid}").exists()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def build_record(
    args: argparse.Namespace,
    run_id: str,
    command: list[str],
    start_wall: datetime,
    end_wall: datetime,
    exit_code: int | None,
    intervals: Iterable[IntervalEnergy],
    monitor: PowerMonitor,
) -> dict[str, object]:
    interval_list = list(intervals)
    totals = {
        "gpu_j": sum(item.gpu_j for item in interval_list),
        "cpu_j": sum(item.cpu_j for item in interval_list),
        "extra_j": sum(item.extra_j for item in interval_list),
        "fixed_j": sum(item.fixed_j for item in interval_list),
        "fallback_j": sum(item.fallback_j for item in interval_list),
    }
    total_j = sum(totals.values())
    duration_s = sum(item.elapsed_s for item in interval_list)
    total_kwh = joules_to_kwh(total_j)
    cost = sum(item.cost_usd for item in interval_list)
    avg_watts = total_j / duration_s if duration_s > 0 else 0.0

    periods: dict[str, dict[str, float]] = {}
    for item in interval_list:
        key = f"{item.rate.season}:{item.rate.period}:{item.rate.tier}"
        bucket = periods.setdefault(key, {"kwh": 0.0, "cost_usd": 0.0, "rate_usd_per_kwh": item.rate.usd_per_kwh})
        bucket["kwh"] += item.total_kwh
        bucket["cost_usd"] += item.cost_usd

    sensor_notes: list[str] = []
    if args.power_source == "fixed":
        sensor_notes.append(f"fixed_watts={args.fixed_watts}")
    else:
        sensor_notes.append("gpu=nvidia-smi" if monitor.used_gpu_sensor else "gpu=unavailable")
        sensor_notes.append("cpu=rapl" if monitor.used_cpu_sensor else "cpu=unavailable")
        if args.extra_watts:
            sensor_notes.append(f"extra_watts={args.extra_watts}")
        if monitor.used_fallback:
            sensor_notes.append(f"fallback_watts={args.fallback_watts}")

    return {
        "run_id": run_id,
        "label": args.label or "",
        "start_time": start_wall.isoformat(),
        "end_time": end_wall.isoformat(),
        "duration_s": round(duration_s, 3),
        "exit_code": exit_code if exit_code is not None else "",
        "total_kwh": round(total_kwh, 9),
        "cost_usd": round(cost, 6),
        "avg_total_watts": round(avg_watts, 3),
        "gpu_kwh": round(joules_to_kwh(totals["gpu_j"]), 9),
        "cpu_kwh": round(joules_to_kwh(totals["cpu_j"]), 9),
        "extra_kwh": round(joules_to_kwh(totals["extra_j"]), 9),
        "fixed_kwh": round(joules_to_kwh(totals["fixed_j"]), 9),
        "fallback_kwh": round(joules_to_kwh(totals["fallback_j"]), 9),
        "rate_mode": "flat" if args.rate_usd_per_kwh is not None else "sdge-tou-dr1-bundled",
        "rate_usd_per_kwh": args.rate_usd_per_kwh if args.rate_usd_per_kwh is not None else "",
        "sdge_tier": args.sdge_tier if args.rate_usd_per_kwh is None else "",
        "sample_interval_s": args.sample_interval,
        "power_source": args.power_source,
        "sensor_notes": "; ".join(sensor_notes),
        "period_breakdown": periods,
        "command": shlex.join(command) if command else f"pid:{args.pid}",
    }


def monitor_until_done(
    args: argparse.Namespace,
    monitor: PowerMonitor,
    proc: subprocess.Popen | None,
) -> tuple[list[IntervalEnergy], int | None]:
    intervals: list[IntervalEnergy] = []
    sample_writer, sample_handle = open_sample_writer(args.samples_out)

    prev = monitor.snapshot()
    exit_code: int | None = None

    try:
        while True:
            if proc is not None:
                try:
                    exit_code = proc.wait(timeout=args.sample_interval)
                    done = True
                except subprocess.TimeoutExpired:
                    done = False
            else:
                time.sleep(args.sample_interval)
                done = not pid_exists(args.pid)

            cur = monitor.snapshot()
            interval = monitor.interval_energy(prev, cur)
            intervals.append(interval)
            write_sample(sample_writer, prev, interval)
            prev = cur

            if done:
                break
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    exit_code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    exit_code = proc.wait()
        raise
    finally:
        if sample_handle:
            sample_handle.close()

    return intervals, exit_code


def list_sensors() -> None:
    gpu = NvidiaSmiReader()
    gpu_watts = gpu.read_total_watts()
    print("Power sensors:")
    if gpu_watts is None:
        detail = f" ({gpu.last_error})" if gpu.last_error else ""
        print(f"  NVIDIA GPU via nvidia-smi: unavailable{detail}")
    else:
        print(f"  NVIDIA GPU via nvidia-smi: {gpu_watts:.2f} W total now")

    rapl = RaplReader()
    if not rapl.domains:
        print("  CPU package via Linux RAPL: unavailable")
    else:
        print("  CPU package via Linux RAPL:")
        for label, path, _max_range_j in rapl.domains:
            print(f"    {label} ({path})")


def print_summary(record: dict[str, object]) -> None:
    print()
    print("Power/cost summary")
    print(f"  run_id:          {record['run_id']}")
    if record["label"]:
        print(f"  label:           {record['label']}")
    print(f"  duration:        {record['duration_s']} s")
    print(f"  energy:          {record['total_kwh']:.9f} kWh")
    print(f"  avg power:       {record['avg_total_watts']} W")
    print(f"  estimated cost:  ${record['cost_usd']:.6f}")
    print(f"  rate mode:       {record['rate_mode']}")
    if record["sdge_tier"]:
        print(f"  SDG&E tier:      {record['sdge_tier']}")
    print(f"  sensors:         {record['sensor_notes']}")
    print(f"  JSONL log:       {record.get('_jsonl_log', '')}")
    print(f"  CSV log:         {record.get('_csv_log', '')}")


def positive_float(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a model run's energy use and estimate electricity cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    parser.add_argument("--label", default="", help="Human-readable run label stored in logs")
    parser.add_argument("--pid", type=int, help="Monitor an existing process ID instead of launching a command")
    parser.add_argument("--sample-interval", type=positive_float, default=1.0, help="Seconds between samples")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Timezone for TOU pricing")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_JSONL_LOG, help="Append detailed run JSONL here")
    parser.add_argument("--csv-log", type=Path, default=DEFAULT_CSV_LOG, help="Append compact CSV run log here")
    parser.add_argument("--samples-out", help="Optional per-sample CSV path")
    parser.add_argument("--list-sensors", action="store_true", help="Print detected power sensors and exit")

    rate = parser.add_argument_group("rate options")
    rate.add_argument(
        "--rate-usd-per-kwh",
        type=positive_float,
        help="Use a flat custom electricity rate instead of SDG&E TOU-DR1",
    )
    rate.add_argument(
        "--sdge-tier",
        choices=("tier1", "tier2"),
        default="tier1",
        help="tier1 applies the up-to-130%% baseline credit; tier2 uses full marginal rates",
    )

    power = parser.add_argument_group("power options")
    power.add_argument(
        "--power-source",
        choices=("sensors", "fixed"),
        default="sensors",
        help="Use live sensors or a fixed whole-system wattage",
    )
    power.add_argument("--fixed-watts", type=positive_float, default=0.0, help="Whole-system watts for --power-source fixed")
    power.add_argument("--extra-watts", type=positive_float, default=0.0, help="Constant watts added to live sensors")
    power.add_argument(
        "--fallback-watts",
        type=positive_float,
        default=0.0,
        help="Constant watts used only when no live GPU/CPU energy is available for an interval",
    )
    power.add_argument("--no-gpu", dest="gpu", action="store_false", help="Disable nvidia-smi GPU power sampling")
    power.add_argument("--no-cpu", dest="cpu", action="store_false", help="Disable Linux RAPL CPU energy sampling")
    power.set_defaults(gpu=True, cpu=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than zero")
    if args.power_source == "fixed" and args.fixed_watts <= 0:
        parser.error("--power-source fixed requires --fixed-watts > 0")

    if args.list_sensors:
        list_sensors()
        return 0

    command = parse_command(args.command)
    if bool(command) == bool(args.pid):
        parser.error("provide exactly one of a command after -- or --pid")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    monitor = PowerMonitor(args)
    start_wall = datetime.now(ZoneInfo(args.timezone))

    proc: subprocess.Popen | None = None
    if command:
        print(f"[power] starting: {shlex.join(command)}", flush=True)
        proc = subprocess.Popen(command)
    else:
        if not pid_exists(args.pid):
            parser.error(f"pid {args.pid} does not exist")
        print(f"[power] monitoring pid {args.pid}", flush=True)

    try:
        intervals, exit_code = monitor_until_done(args, monitor, proc)
    except KeyboardInterrupt:
        end_wall = datetime.now(ZoneInfo(args.timezone))
        print(f"\n[power] interrupted at {end_wall.isoformat()}", file=sys.stderr)
        return 130

    end_wall = datetime.now(ZoneInfo(args.timezone))
    record = build_record(args, run_id, command, start_wall, end_wall, exit_code, intervals, monitor)
    record["_jsonl_log"] = str(args.log_path)
    record["_csv_log"] = str(args.csv_log)

    append_jsonl(args.log_path, {k: v for k, v in record.items() if not k.startswith("_")})
    append_csv(args.csv_log, record)
    print_summary(record)

    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
