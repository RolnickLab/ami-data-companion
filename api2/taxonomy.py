from dataclasses import dataclass
from typing import Optional


@dataclass
class Taxon:
    gbif_id: int
    name: Optional[str]
    genus: Optional[str]
    family: Optional[str]
    source: Optional[str]


def fetch_gbif_species(gbif_id: int) -> Optional[Taxon]:
    """
    Look up taxon name from GBIF API. Cache results in user_data_path.
    """

    logger.info(f"Looking up species name for GBIF id {gbif_id}")
    base_url = "https://api.gbif.org/v1/species/{gbif_id}"
    url = base_url.format(gbif_id=gbif_id)

    try:
        taxon_data = get_or_download_file(
            url, destination_dir=get_user_data_dir(), prefix="taxa/gbif", suffix=".json"
        )
        data: dict = json.load(taxon_data.open())
    except urllib.error.HTTPError:
        logger.warn(f"Could not find species with gbif_id {gbif_id} in {url}")
        return None
    except json.decoder.JSONDecodeError:
        logger.warn(f"Could not parse JSON response from {url}")
        return None

    taxon = Taxon(
        gbif_id=gbif_id,
        name=data["canonicalName"],
        genus=data["genus"],
        family=data["family"],
        source="gbif",
    )
    return taxon


def lookup_gbif_species(species_list_path: str, gbif_id: int) -> Taxon:
    """
    Look up taxa names from a Darwin Core Archive file (DwC-A).

    Example:
    https://docs.google.com/spreadsheets/d/1E3-GAB0PSKrnproAC44whigMvnAkbkwUmwXUHMKMOII/edit#gid=1916842176

    @TODO Optionally look up species name from GBIF API
    Example https://api.gbif.org/v1/species/5231190
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for this function")

    local_path = get_or_download_file(
        species_list_path, destination_dir=get_user_data_dir(), prefix="taxa"
    )
    df = pd.read_csv(local_path)
    taxon = None
    # look up single row by gbif_id
    try:
        row = df.loc[df["taxon_key_gbif_id"] == gbif_id].iloc[0]
    except IndexError:
        logger.warn(
            f"Could not find species with gbif_id {gbif_id} in {species_list_path}"
        )
    else:
        taxon = Taxon(
            gbif_id=gbif_id,
            name=row["search_species_name"],
            genus=row["genus_name"],
            family=row["family_name"],
            source=row["source"],
        )

    if not taxon:
        taxon = fetch_gbif_species(gbif_id)

    if not taxon:
        return Taxon(
            gbif_id=gbif_id, name=str(gbif_id), genus=None, family=None, source=None
        )

    return taxon


def replace_gbif_id_with_name(name) -> str:
    """
    If the name appears to be a GBIF ID, then look up the species name from GBIF.
    """
    try:
        gbif_id = int(name)
    except ValueError:
        return name
    else:
        taxon = fetch_gbif_species(gbif_id)
        if taxon and taxon.name:
            return taxon.name
        else:
            return name
