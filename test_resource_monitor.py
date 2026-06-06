from resource_monitor import (
    cgroup_cpu_metrics_between,
    cpu_percent_between,
    parse_cpuset_count,
    read_cgroup_cpu_max,
    parse_nvidia_smi_query,
    read_loadavg,
    read_meminfo,
    read_proc_stat,
    resource_tensorboard_scalars,
    usable_cpu_count,
)


def test_cpu_percent_between_proc_stat_samples():
    previous = {"idle": 100, "total": 200}
    current = {"idle": 125, "total": 300}

    assert cpu_percent_between(previous, current) == 75.0


def test_parse_cpuset_count_ranges_and_singletons():
    assert parse_cpuset_count("0-3,8,10-11") == 7


def test_read_cgroup_cpu_max_reports_quota_cores(tmp_path):
    cpu_max_path = tmp_path / "cpu.max"
    cpu_max_path.write_text("1152000 100000\n", encoding="utf-8")

    cpu_max = read_cgroup_cpu_max(cpu_max_path)

    assert cpu_max["quota_usec"] == 1152000
    assert cpu_max["period_usec"] == 100000
    assert cpu_max["quota_cores"] == 11.52
    assert cpu_max["unlimited"] is False


def test_cgroup_cpu_metrics_use_allocated_capacity():
    previous = {
        "usage_usec": 1_000_000,
        "nr_periods": 10,
        "nr_throttled": 2,
        "throttled_usec": 200_000,
    }
    current = {
        "usage_usec": 7_000_000,
        "nr_periods": 20,
        "nr_throttled": 5,
        "throttled_usec": 800_000,
    }

    metrics = cgroup_cpu_metrics_between(
        previous,
        current,
        elapsed_s=2.0,
        capacity_cores=12.0,
    )

    assert metrics["cpu_used_cores"] == 3.0
    assert metrics["cpu_util_percent"] == 25.0
    assert metrics["cpu_throttled_usec_delta"] == 600_000
    assert metrics["cpu_throttled_periods_delta"] == 3
    assert metrics["cpu_throttled_period_fraction"] == 0.3


def test_usable_cpu_count_ceilings_cgroup_capacity(monkeypatch):
    import resource_monitor

    monkeypatch.setattr(
        resource_monitor,
        "read_cgroup_cpu_max",
        lambda: {"quota_cores": 11.52},
    )
    monkeypatch.setattr(resource_monitor, "read_cpuset_count", lambda: 72)
    monkeypatch.setattr(resource_monitor.os, "cpu_count", lambda: 72)

    assert usable_cpu_count() == 12


def test_resource_tensorboard_scalars_include_cgroup_cpu_and_gpu():
    sample = {
        "cpu_util_percent": 30.0,
        "cpu_used_cores": 3.45,
        "host_cpu_util_percent": 90.0,
        "cpu_throttled_period_fraction": 0.2,
        "cpu": {
            "capacity_cores": 11.52,
            "quota_cores": 11.52,
            "usable_workers": 12,
        },
        "memory": {
            "memory_used_percent": 10.0,
            "memory_used_mb": 14000.0,
        },
        "gpu": {
            "gpu_util_percent": 71,
            "gpu_memory_used_mb": 1236,
            "gpu_temperature_c": 51,
        },
    }

    scalars = resource_tensorboard_scalars(sample)

    assert scalars["resource/cpu_util_percent"] == 30.0
    assert scalars["resource/cpu_used_cores"] == 3.45
    assert scalars["resource/host_cpu_util_percent"] == 90.0
    assert scalars["resource/cpu_throttled_period_fraction"] == 0.2
    assert scalars["resource/cpu_capacity_cores"] == 11.52
    assert scalars["resource/cpu_usable_workers"] == 12.0
    assert scalars["resource/memory_used_percent"] == 10.0
    assert scalars["resource/gpu_util_percent"] == 71.0
    assert scalars["resource/gpu_memory_used_mb"] == 1236.0


def test_read_proc_stat_parses_cpu_totals(tmp_path):
    stat_path = tmp_path / "stat"
    stat_path.write_text("cpu  10 20 30 40 5 6 7 8 9 10\n", encoding="utf-8")

    totals = read_proc_stat(stat_path)

    assert totals["idle"] == 45
    assert totals["total"] == 145


def test_read_loadavg_parses_loads(tmp_path):
    loadavg_path = tmp_path / "loadavg"
    loadavg_path.write_text("1.25 2.50 3.75 1/100 12345\n", encoding="utf-8")

    load = read_loadavg(loadavg_path)

    assert load == {"load1": 1.25, "load5": 2.5, "load15": 3.75}


def test_read_meminfo_reports_used_memory(tmp_path):
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(
        "MemTotal:       1048576 kB\n"
        "MemAvailable:    262144 kB\n",
        encoding="utf-8",
    )

    memory = read_meminfo(meminfo_path)

    assert memory["memory_total_mb"] == 1024
    assert memory["memory_available_mb"] == 256
    assert memory["memory_used_mb"] == 768
    assert memory["memory_used_percent"] == 75.0


def test_parse_nvidia_smi_query_csv():
    gpu = parse_nvidia_smi_query("71, 1236, 8188, 51\n")

    assert gpu == {
        "gpu_util_percent": 71,
        "gpu_memory_used_mb": 1236,
        "gpu_memory_total_mb": 8188,
        "gpu_temperature_c": 51,
    }
