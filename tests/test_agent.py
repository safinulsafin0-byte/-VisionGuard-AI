"""Tests for VisionAgent multi-step evidence-driven workflow."""

import pytest
import numpy as np

from visionguard.agent import VisionAgent
from visionguard.synthetic_data import (
    generate_blurred_checkerboard,
    generate_sharp_checkerboard,
    generate_surface_with_artifact,
    generate_underexposed_pattern,
)


def test_agent_nominal_workflow():
    agent = VisionAgent(max_steps=5)
    img = generate_sharp_checkerboard(256, 256)
    session = agent.inspect(img)

    assert session.total_steps <= 5
    assert len(session.action_trace) >= 2
    # First step should always be survey_global_image
    assert session.action_trace[0].tool_name == "survey_global_image"
    # Final step should be conclude_inspection
    assert session.action_trace[-1].tool_name == "conclude_inspection"
    assert session.final_verdict.verdict_type == "NOMINAL_CLEAR"


def test_agent_triggers_sharpness_grid_on_blurred_input():
    agent = VisionAgent(max_steps=5)
    img = generate_blurred_checkerboard(256, 256, blur_kernel=25)
    session = agent.inspect(img)

    tools_used = [a.tool_name for a in session.action_trace]
    assert "compute_local_sharpness_grid" in tools_used
    assert session.final_verdict.verdict_type == "IMAGE_QUALITY_DEFECT"


def test_agent_triggers_crop_reinspection_on_artifact():
    agent = VisionAgent(max_steps=5)
    img, _ = generate_surface_with_artifact(512, 512)
    session = agent.inspect(img)

    tools_used = [a.tool_name for a in session.action_trace]
    assert "reinspect_candidate_crop" in tools_used
    assert session.final_verdict.verdict_type == "ANOMALOUS_CANDIDATE_FLAGGED"
    assert len(session.target_crops_inspected) > 0


def test_agent_triggers_regional_exposure_on_underexposed():
    agent = VisionAgent(max_steps=5)
    img = generate_underexposed_pattern(256, 256, mean_target=18)
    session = agent.inspect(img)

    tools_used = [a.tool_name for a in session.action_trace]
    assert "analyze_regional_exposure" in tools_used
    assert session.final_verdict.verdict_type == "IMAGE_QUALITY_DEFECT"


def test_agent_step_cap():
    # Force max_steps to 2
    agent = VisionAgent(max_steps=2)
    img, _ = generate_surface_with_artifact(256, 256)
    session = agent.inspect(img)
    # Total steps cannot exceed max_steps + 1 (conclusion step)
    assert session.total_steps <= 3

