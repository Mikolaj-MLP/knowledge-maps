import argparse
import sys
from collections.abc import Sequence

from knowledge_maps.bootstrap import create_service
from knowledge_maps.errors import KnowledgeMapsError


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)

    try:
        knowledge_map = create_service().build(arguments.arxiv_id_or_url)
    except (KnowledgeMapsError, ValueError) as error:
        print(f"knowledge-maps: {error}", file=sys.stderr)
        return 1

    # ASCII escapes keep redirected JSON valid on Windows consoles with legacy encodings.
    print(knowledge_map.model_dump_json(indent=2, ensure_ascii=True))
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge-maps",
        description="Build a prerequisite graph for an arXiv paper.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    build_parser = subcommands.add_parser("build", help="Build a prerequisite graph.")
    build_parser.add_argument("arxiv_id_or_url", help="An arXiv ID or URL.")
    return parser
