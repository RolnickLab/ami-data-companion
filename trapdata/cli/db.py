from datetime import datetime
from typing import List, Tuple, TypedDict

import sqlalchemy as sa
import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from trapdata import db
from trapdata.cli import settings
from trapdata.db.base import get_session
from trapdata.db.models.images import TrapImage

cli = typer.Typer(no_args_is_help=True)
console = Console()


class DuplicateGroup(TypedDict):
    path: str
    total_count: int
    keep_record: TrapImage
    delete_records: List[TrapImage]
    delete_count: int


def find_duplicate_images(session: sa.orm.Session) -> List[DuplicateGroup]:
    """
    Find duplicate TrapImage records based on matching path values.

    OPTIMIZED VERSION: Uses a single efficient query with window functions
    instead of N+1 queries to avoid performance bottleneck.

    Returns a list of DuplicateGroup objects containing information about
    each set of duplicates, including which record to keep and which to delete.
    """
    # Single optimized query: get all images that have duplicates, ordered properly
    # Uses a window function to count duplicates and only return rows where count > 1
    # Note: We use options to avoid eager loading issues with relationships
    duplicate_images_query = (
        sa.select(
            TrapImage,
            sa.func.count(TrapImage.id)
            .over(partition_by=TrapImage.path)
            .label("path_count"),
        )
        .options(sa.orm.noload("*"))  # Disable eager loading to avoid unique() issues
        .order_by(
            TrapImage.path,
            sa.asc(TrapImage.timestamp).nulls_last(),
            sa.asc(TrapImage.id),
        )
    )

    # Execute query and get all results - unique() handles duplicate rows from joins
    results = session.execute(duplicate_images_query).unique().all()

    # Filter to only duplicates and group by path
    duplicate_groups = []
    current_group = []
    current_path = None

    for row in results:
        image, path_count = row

        # Only process images that have duplicates
        if path_count <= 1:
            continue

        # Start new group if path changed
        if current_path != image.path:
            # Process previous group if it exists
            if current_group and len(current_group) > 1:
                duplicate_groups.append(_create_duplicate_group(current_group))

            # Start new group
            current_group = [image]
            current_path = image.path
        else:
            current_group.append(image)

    # Process final group
    if current_group and len(current_group) > 1:
        duplicate_groups.append(_create_duplicate_group(current_group))

    return duplicate_groups


def _create_duplicate_group(duplicate_images: List[TrapImage]) -> DuplicateGroup:
    """Helper function to create a DuplicateGroup from a list of duplicate images."""
    keep_record = get_retention_candidate(duplicate_images)
    delete_records = [img for img in duplicate_images if img.id != keep_record.id]

    return DuplicateGroup(
        path=duplicate_images[0].path,
        total_count=len(duplicate_images),
        keep_record=keep_record,
        delete_records=delete_records,
        delete_count=len(delete_records),
    )


def get_retention_candidate(duplicates: List[TrapImage]) -> TrapImage:
    """
    Determine which TrapImage record to keep from a list of duplicates.

    Selection criteria (in order of precedence):
    1. Oldest timestamp (non-null timestamps first)
    2. Lowest ID as fallback for same/null timestamps
    """
    if not duplicates:
        raise ValueError("Cannot determine retention candidate from empty list")

    if len(duplicates) == 1:
        return duplicates[0]

    # Sort by timestamp (oldest first, nulls last), then by ID (lowest first)
    sorted_duplicates = sorted(
        duplicates,
        key=lambda img: (
            img.timestamp is not None,  # Non-null timestamps first
            (
                img.timestamp if img.timestamp else datetime.max
            ),  # Then by timestamp (oldest first)
            img.id,  # Finally by ID (lowest first)
        ),
        reverse=False,  # Changed to False to get oldest first
    )

    return sorted_duplicates[0]


def execute_duplicate_removal(
    session: sa.orm.Session, duplicate_groups: List[DuplicateGroup]
) -> Tuple[int, int]:
    """
    Execute deletion of duplicate records with transaction safety.

    Args:
        session: Database session for performing deletions
        duplicate_groups: List of duplicate groups to process

    Returns:
        Tuple of (groups_processed, records_deleted)

    Raises:
        Exception: Re-raises any database errors after rollback
    """
    groups_processed = 0
    records_deleted = 0

    try:
        for group in duplicate_groups:
            # Delete each duplicate record in this group
            for record in group["delete_records"]:
                session.delete(record)
                records_deleted += 1

            groups_processed += 1

        # Commit all deletions at once
        session.commit()

    except Exception as e:
        # Rollback transaction on any error
        session.rollback()
        console.print(f"[red]Error during deletion: {e}[/red]")
        console.print(
            "[yellow]Database transaction rolled back. No changes were made.[/yellow]"
        )
        raise

    return groups_processed, records_deleted


def display_duplicate_summary(duplicate_groups: List[DuplicateGroup]) -> None:
    """
    Display a formatted summary of duplicate groups using Rich console output.
    """
    if not duplicate_groups:
        console.print("[green]No duplicate images found in the database.[/green]")
        return

    total_duplicates = sum(group["delete_count"] for group in duplicate_groups)
    total_groups = len(duplicate_groups)

    console.print(
        f"\n[yellow]Found {total_groups} duplicate path(s) with {total_duplicates} records to remove.[/yellow]"
    )

    # Create summary table
    table = Table(
        title="Duplicate Images Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Path", style="cyan", no_wrap=False, width=40)
    table.add_column("Total", justify="center", style="white")
    table.add_column("Keep ID", justify="center", style="green")
    table.add_column("Keep Timestamp", style="green", no_wrap=False, width=20)
    table.add_column("Delete IDs", justify="center", style="red")

    for group in duplicate_groups:
        keep_timestamp = (
            group["keep_record"].timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if group["keep_record"].timestamp
            else "No timestamp"
        )
        delete_ids = ", ".join(str(record.id) for record in group["delete_records"])

        table.add_row(
            group["path"],
            str(group["total_count"]),
            str(group["keep_record"].id),
            keep_timestamp,
            delete_ids,
        )

    console.print(table)


def confirm_deletion(total_deletions: int) -> bool:
    """
    Prompt user for confirmation before executing deletions.

    Args:
        total_deletions: Number of records to be deleted

    Returns:
        True if user confirms deletion, False otherwise
    """
    console.print(
        f"\n[red]WARNING: This will permanently delete {total_deletions} TrapImage records.[/red]"
    )
    console.print("[yellow]This action cannot be undone.[/yellow]")

    return Confirm.ask(
        "Are you sure you want to proceed with the deletion?", default=False
    )


# Module-level option definitions to avoid B008 linting error
DRY_RUN_OPTION = typer.Option(
    True,
    "--dry-run/--execute",
    help="Show what would be deleted without actually deleting (default: dry-run)",
)
NO_CONFIRM_OPTION = typer.Option(
    False, "--no-confirm", help="Skip confirmation prompt when executing deletions"
)


@cli.command("remove-duplicate-images")
def remove_duplicates_command(
    dry_run: bool = DRY_RUN_OPTION,
    confirm: bool = NO_CONFIRM_OPTION,
) -> None:
    """
    Find and remove duplicate TrapImage entries from the database.

    Identifies TrapImage records with identical path values and removes duplicates,
    keeping only the oldest record (by timestamp, with ID as fallback).

    By default runs in dry-run mode to preview changes. Use --execute to perform actual
    deletions.
    """
    console.print(
        "[bold blue]Scanning database for duplicate TrapImage records...[/bold blue]"
    )

    try:
        with get_session(settings.database_url) as session:
            # Find duplicate image groups
            duplicate_groups = find_duplicate_images(session)

            # Display summary
            display_duplicate_summary(duplicate_groups)

            if not duplicate_groups:
                return

            total_deletions = sum(group["delete_count"] for group in duplicate_groups)

            if dry_run:
                console.print(
                    f"\n[cyan]DRY RUN: {total_deletions} records would be deleted.[/cyan]"
                )
                console.print("[cyan]Use --execute to perform actual deletions.[/cyan]")
                return

            # Execute mode - ask for confirmation unless skipped
            if not confirm:
                if not confirm_deletion(total_deletions):
                    console.print("[yellow]Deletion cancelled by user.[/yellow]")
                    return

            # Perform deletions
            console.print("[yellow]Executing deletions...[/yellow]")
            groups_processed, records_deleted = execute_duplicate_removal(
                session, duplicate_groups
            )

            console.print(
                f"[green]Successfully processed {groups_processed} duplicate groups.[/green]"
            )
            console.print(
                f"[green]Deleted {records_deleted} duplicate records.[/green]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@cli.command()
def create():
    """
    Create database tables and sqlite file if neccessary.
    """
    db.create_db(settings.database_url)
    db.migrate(settings.database_url)
    db.check_db(settings.database_url, quiet=False)


@cli.command()
def update():
    """
    Run database migrations to update the database schema.
    """
    db.migrate(settings.database_url)
    db.check_db(settings.database_url, quiet=False)


@cli.command()
def reset():
    """
    Backup and recreate database tables.
    """
    reset = typer.confirm("Are you sure you want to reset the database?")
    if reset:
        db.reset_db(settings.database_url)
    else:
        typer.Abort()


@cli.command()
def check():
    """
    Validate database tables and ORM models.
    """
    db.check_db(settings.database_url, quiet=False)


if __name__ == "__main__":
    cli()
