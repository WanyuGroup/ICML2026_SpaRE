from __future__ import annotations

from spare_molgen.cli import PUBLIC_COMMANDS, build_parser


def test_cli_exposes_only_six_public_workflow_commands():
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")

    assert tuple(subparsers_action.choices) == PUBLIC_COMMANDS
