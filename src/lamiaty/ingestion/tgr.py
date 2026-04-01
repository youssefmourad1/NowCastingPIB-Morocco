"""
TGR (Trésorerie Générale du Royaume) data scraper — Phase 2 stub.

Will ingest monthly budget execution data (Investissement_Etat) from
finances.gov.ma/TGR once the series definition is confirmed with TGR/MEF.

IMPORTANT: Investissement_Etat is currently excluded from the DFM
(include_in_model: false) pending definition confirmation. This scraper
should only be activated after that confirmation.
"""


def fetch_budget_execution():
    """Fetch budget execution data from TGR. Phase 2 implementation."""
    raise NotImplementedError("TGR scraper implemented in Phase 2.")
