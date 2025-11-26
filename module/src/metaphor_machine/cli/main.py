"""
Command-line interface for the Metaphor Machine.

Provides commands for generating metaphors, chains, and batches
with full control over seeds, diversity, and output formats.

Usage:
    metaphor generate --seed 42 --count 5
    metaphor chain --seed 42 --format suno
    metaphor batch --count 10 --min-distance 3 --output results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

import click
from rich.console import Console
from rich.table import Table

from metaphor_machine import __version__
from metaphor_machine.core.generator import MetaphorGenerator
from metaphor_machine.core.metaphor import batch_min_distance
from metaphor_machine.schemas.components import StyleComponents
from metaphor_machine.schemas.config import (
    ChainSeparator,
    DiversityConfig,
    GeneratorConfig,
    OutputFormat,
)

console = Console()


def get_default_components_path() -> Path:
    """Find the default style_components.yaml location."""
    # Check common locations
    candidates = [
        Path("style_components.yaml"),
        Path("data/style_components.yaml"),
        Path.home() / ".metaphor-machine" / "style_components.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Return first candidate (will raise FileNotFoundError if used)
    return candidates[0]


@click.group()
@click.version_option(version=__version__, prog_name="metaphor-machine")
@click.option(
    "--components",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to style_components.yaml",
)
@click.pass_context
def cli(ctx: click.Context, components: Path | None) -> None:
    """
    Metaphor Machine: Generate rich style descriptions for AI music systems.

    The Metaphor Machine produces structured 5-slot "metaphors" that guide
    AI music generators (Suno, Producer.ai) toward specific textures,
    dynamics, and emotional arcs.

    \b
    Examples:
        metaphor generate                    # Single random metaphor
        metaphor generate --seed 42          # Reproducible generation
        metaphor chain --format suno         # 3-act chain for Suno
        metaphor batch --count 10            # Batch with diversity
    """
    ctx.ensure_object(dict)

    # Load components
    components_path = components or get_default_components_path()
    try:
        ctx.obj["components"] = StyleComponents.from_yaml(components_path)
        ctx.obj["components_path"] = components_path
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] Could not find style_components.yaml at {components_path}",
            highlight=False,
        )
        console.print(
            "Use --components to specify the path, or create the file.",
            style="dim",
        )
        ctx.exit(1)


@cli.command()
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
@click.option("--count", "-n", type=int, default=1, help="Number of metaphors to generate")
@click.option("--genre-hint", "-g", type=str, default=None, help="Genre to bias toward")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json", "suno"]),
    default="plain",
    help="Output format",
)
@click.option("--output", "-o", type=click.File("w"), default="-", help="Output file (default: stdout)")
@click.pass_context
def generate(
    ctx: click.Context,
    seed: int | None,
    count: int,
    genre_hint: str | None,
    output_format: str,
    output: TextIO,
) -> None:
    """
    Generate single 5-slot metaphors.

    Each metaphor contains: genre anchor, intimate gesture, dynamic tension,
    sensory bridge, and emotional anchor.

    \b
    Examples:
        metaphor generate
        metaphor generate --seed 42 --count 5
        metaphor generate --genre-hint darkwave --format json
    """
    components: StyleComponents = ctx.obj["components"]

    config = GeneratorConfig(seed=seed, genre_hint=genre_hint)
    generator = MetaphorGenerator(components, config=config)

    metaphors = [generator.generate_single() for _ in range(count)]

    if output_format == "json":
        data = {
            "seed": seed,
            "count": count,
            "genre_hint": genre_hint,
            "metaphors": [m.to_dict() for m in metaphors],
        }
        output.write(json.dumps(data, indent=2))
        output.write("\n")
    elif output_format == "plain":
        for m in metaphors:
            output.write(str(m))
            output.write("\n")
    else:  # suno
        for m in metaphors:
            output.write(str(m))
            output.write("\n")

    if output == sys.stdout and seed is not None:
        console.print(f"\n[dim]Seed: {seed}[/dim]", highlight=False)


@cli.command()
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
@click.option("--count", "-n", type=int, default=1, help="Number of chains to generate")
@click.option("--genre-hint", "-g", type=str, default=None, help="Genre to bias toward")
@click.option(
    "--separator",
    type=click.Choice(["arrow", "semicolon", "newline", "pipe"]),
    default="arrow",
    help="Separator between chain parts",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json", "suno"]),
    default="suno",
    help="Output format",
)
@click.option("--output", "-o", type=click.File("w"), default="-", help="Output file")
@click.pass_context
def chain(
    ctx: click.Context,
    seed: int | None,
    count: int,
    genre_hint: str | None,
    separator: str,
    output_format: str,
    output: TextIO,
) -> None:
    """
    Generate 3-act metaphor chains (Intro → Mid → Outro).

    Each chain describes a complete track arc with distinct metaphors
    for opening, peak, and resolution phases.

    \b
    Examples:
        metaphor chain
        metaphor chain --seed 42 --separator newline
        metaphor chain --count 3 --format json
    """
    components: StyleComponents = ctx.obj["components"]

    separator_map = {
        "arrow": ChainSeparator.ARROW,
        "semicolon": ChainSeparator.SEMICOLON,
        "newline": ChainSeparator.NEWLINE,
        "pipe": ChainSeparator.PIPE,
    }

    config = GeneratorConfig(
        seed=seed,
        genre_hint=genre_hint,
        chain_separator=separator_map[separator],
    )
    generator = MetaphorGenerator(components, config=config)

    chains = [generator.generate_chain() for _ in range(count)]

    if output_format == "json":
        data = {
            "seed": seed,
            "count": count,
            "genre_hint": genre_hint,
            "chains": [c.to_dict() for c in chains],
        }
        output.write(json.dumps(data, indent=2))
        output.write("\n")
    else:
        sep = separator_map[separator].value
        for c in chains:
            output.write(c.to_suno_style(separator=sep))
            output.write("\n")
            if count > 1:
                output.write("\n")  # Extra line between chains

    if output == sys.stdout and seed is not None:
        console.print(f"\n[dim]Seed: {seed}[/dim]", highlight=False)


@cli.command()
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
@click.option("--count", "-n", type=int, default=10, help="Number of metaphors to generate")
@click.option("--min-distance", "-d", type=int, default=3, help="Minimum Hamming distance (0-5)")
@click.option("--max-retries", type=int, default=100, help="Max retries per metaphor")
@click.option("--genre-hint", "-g", type=str, default=None, help="Genre to bias toward")
@click.option("--chains/--singles", default=False, help="Generate chains instead of singles")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plain", "json", "table"]),
    default="plain",
    help="Output format",
)
@click.option("--output", "-o", type=click.File("w"), default="-", help="Output file")
@click.pass_context
def batch(
    ctx: click.Context,
    seed: int | None,
    count: int,
    min_distance: int,
    max_retries: int,
    genre_hint: str | None,
    chains: bool,
    output_format: str,
    output: TextIO,
) -> None:
    """
    Generate a diverse batch of metaphors or chains.

    Enforces minimum Hamming distance between generated items to ensure
    variety. Distance of 3 means at least 3 of 5 slots must differ.

    \b
    Examples:
        metaphor batch --count 10 --min-distance 3
        metaphor batch --chains --count 5
        metaphor batch --seed 42 --format table
    """
    components: StyleComponents = ctx.obj["components"]

    diversity = DiversityConfig(
        min_hamming_distance=min_distance,
        max_retries=max_retries,
        allow_partial=True,
    )
    config = GeneratorConfig(seed=seed, genre_hint=genre_hint, diversity=diversity)
    generator = MetaphorGenerator(components, config=config)

    if chains:
        results = generator.generate_chain_batch(count, enforce_diversity=True)
        actual_count = len(results)

        if output_format == "json":
            data = {
                "seed": seed,
                "requested_count": count,
                "actual_count": actual_count,
                "min_distance": min_distance,
                "chains": [c.to_dict() for c in results],
            }
            output.write(json.dumps(data, indent=2))
            output.write("\n")
        elif output_format == "table":
            table = Table(title=f"Generated Chains (seed={seed})")
            table.add_column("#", style="dim")
            table.add_column("Intro")
            table.add_column("Mid")
            table.add_column("Outro")
            for i, c in enumerate(results, 1):
                table.add_row(
                    str(i),
                    str(c.intro)[:50] + "...",
                    str(c.mid)[:50] + "...",
                    str(c.outro)[:50] + "...",
                )
            console.print(table)
        else:
            for c in results:
                output.write(c.to_suno_style())
                output.write("\n\n")
    else:
        results = generator.generate_batch(count, enforce_diversity=True)
        actual_count = len(results)
        min_dist = batch_min_distance(results) if len(results) > 1 else 0

        if output_format == "json":
            data = {
                "seed": seed,
                "requested_count": count,
                "actual_count": actual_count,
                "min_distance_requested": min_distance,
                "min_distance_achieved": min_dist,
                "metaphors": [m.to_dict() for m in results],
            }
            output.write(json.dumps(data, indent=2))
            output.write("\n")
        elif output_format == "table":
            table = Table(title=f"Generated Metaphors (seed={seed})")
            table.add_column("#", style="dim")
            table.add_column("Genre")
            table.add_column("Gesture")
            table.add_column("Tension")
            table.add_column("Bridge")
            table.add_column("Anchor")
            for i, m in enumerate(results, 1):
                table.add_row(
                    str(i),
                    m.genre_anchor.value[:20],
                    m.intimate_gesture.value[:20],
                    m.dynamic_tension.value[:20],
                    m.sensory_bridge.value[:20],
                    m.emotional_anchor.value[:20],
                )
            console.print(table)
        else:
            for m in results:
                output.write(str(m))
                output.write("\n")

    if output == sys.stdout:
        if actual_count < count:
            console.print(
                f"\n[yellow]Warning:[/yellow] Only generated {actual_count}/{count} "
                f"(diversity constraints)",
                highlight=False,
            )
        if seed is not None:
            console.print(f"[dim]Seed: {seed}[/dim]", highlight=False)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """
    Display information about loaded style components.

    Shows pool sizes and estimated combinatorial space.
    """
    components: StyleComponents = ctx.obj["components"]
    components_path: Path = ctx.obj["components_path"]

    console.print(f"\n[bold]Style Components:[/bold] {components_path}")
    console.print()

    table = Table(title="Pool Sizes")
    table.add_column("Pool", style="cyan")
    table.add_column("Count", justify="right")

    for pool_name, count in components.get_pool_sizes().items():
        table.add_row(pool_name, str(count))

    console.print(table)

    space = components.estimate_combinatorial_space()
    console.print(f"\n[bold]Estimated combinatorial space:[/bold] {space:,} unique metaphors")
    console.print()


@cli.command()
@click.option("--seed", "-s", type=int, required=True, help="Seed to explore")
@click.option("--range", "-r", "seed_range", type=int, default=10, help="Seeds to show around target")
@click.pass_context
def explore(ctx: click.Context, seed: int, seed_range: int) -> None:
    """
    Explore seeds around a known good value.

    Useful for finding variations of a promising seed.

    \b
    Example:
        metaphor explore --seed 42 --range 5
    """
    components: StyleComponents = ctx.obj["components"]

    console.print(f"\n[bold]Exploring seeds {seed - seed_range} to {seed + seed_range}[/bold]\n")

    table = Table()
    table.add_column("Seed", style="cyan", justify="right")
    table.add_column("Generated Metaphor")

    for s in range(seed - seed_range, seed + seed_range + 1):
        generator = MetaphorGenerator(components, seed=s)
        m = generator.generate_single()
        style = "bold green" if s == seed else ""
        table.add_row(str(s), str(m), style=style)

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
