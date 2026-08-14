from __future__ import annotations

import io
import logging
from datetime import datetime

import pytest

from ik_benchmark.log_config import (
    PACKAGE_LOGGER_NAME,
    configure_logging,
)
from ik_benchmark.paths import (
    make_timestamped_run_dir,
    resolve_log_file,
)


def test_configure_logging_writes_package_logs_to_stream():
    stream = io.StringIO()
    logger = configure_logging(level="INFO", stream=stream)

    logger.info("benchmark started")

    output = stream.getvalue()
    assert "INFO" in output
    assert PACKAGE_LOGGER_NAME in output
    assert "benchmark started" in output


def test_configure_logging_respects_level():
    stream = io.StringIO()
    logger = configure_logging(level="WARNING", stream=stream)

    logger.info("hidden")
    logger.warning("visible")

    output = stream.getvalue()
    assert "hidden" not in output
    assert "visible" in output


def test_configure_logging_can_write_file_when_quiet(tmp_path):
    log_file = tmp_path / "logs" / "benchmark.log"
    stream = io.StringIO()
    logger = configure_logging(
        level="INFO",
        log_file=log_file,
        quiet=True,
        stream=stream,
    )

    logger.info("file only")

    assert stream.getvalue() == ""
    assert "file only" in log_file.read_text()


def test_configure_logging_overwrites_file_by_default(tmp_path):
    log_file = tmp_path / "benchmark.log"
    log_file.write_text("old log\n")
    logger = configure_logging(level="INFO", log_file=log_file, quiet=True)

    logger.info("new log")

    text = log_file.read_text()
    assert "new log" in text
    assert "old log" not in text


def test_configure_logging_can_append_to_file(tmp_path):
    log_file = tmp_path / "benchmark.log"
    log_file.write_text("old log\n")
    logger = configure_logging(level="INFO", log_file=log_file, mode="a", quiet=True)

    logger.info("new log")

    text = log_file.read_text()
    assert "old log" in text
    assert "new log" in text


def test_configure_logging_is_idempotent():
    stream = io.StringIO()
    logger = configure_logging(level="INFO", stream=stream)
    logger = configure_logging(level="INFO", stream=stream)

    logger.info("once")

    assert stream.getvalue().count("once") == 1


def test_configure_logging_captures_package_child_loggers():
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream)
    logger = logging.getLogger("ik_benchmark.runner")

    logger.info("child module log")

    output = stream.getvalue()
    assert "ik_benchmark.runner" in output
    assert "child module log" in output


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError, match="log level"):
        configure_logging(level="loud")


def test_configure_logging_rejects_invalid_file_mode():
    with pytest.raises(ValueError, match="log mode"):
        configure_logging(mode="x")


def test_make_timestamped_run_dir_creates_run_dir(tmp_path):
    run_dir = make_timestamped_run_dir(
        tmp_path,
        now=datetime(2026, 6, 23, 15, 30, 45),
    )

    assert run_dir == tmp_path / "runs" / "2026-06-23_153045"
    assert run_dir.is_dir()


def test_make_timestamped_run_dir_avoids_existing_dir(tmp_path):
    existing = tmp_path / "runs" / "2026-06-23_153045"
    existing.mkdir(parents=True)

    run_dir = make_timestamped_run_dir(
        tmp_path,
        now=datetime(2026, 6, 23, 15, 30, 45),
    )

    assert run_dir == tmp_path / "runs" / "2026-06-23_153045_02"
    assert run_dir.is_dir()


def test_resolve_log_file_places_relative_path_in_base_dir(tmp_path):
    assert resolve_log_file("benchmark.log", tmp_path) == tmp_path / "benchmark.log"


def test_resolve_log_file_keeps_absolute_path(tmp_path):
    log_file = tmp_path / "custom.log"

    assert resolve_log_file(log_file, tmp_path / "run") == log_file


def test_setup_benchmark_run_configures_log_file_under_run_dir(tmp_path):
    from scripts.run_benchmark import setup_benchmark_run

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "logging:",
                "  level: INFO",
                "  file: benchmark.log",
                "  mode: w",
                "  quiet: true",
            ]
        )
    )

    cfg, run_dir = setup_benchmark_run(config_path=config_path, out_dir=tmp_path)
    log_file = run_dir / "benchmark.log"

    assert cfg["logging"]["file"] == "benchmark.log"
    assert run_dir.parent == tmp_path / "runs"
    assert log_file.is_file()
    assert "benchmark run configured" in log_file.read_text()


@pytest.fixture(autouse=True)
def reset_package_logging():
    yield
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
