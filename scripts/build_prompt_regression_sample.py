"""Build a fixed relation-prompt regression set from the Qwen candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple


class Case(NamedTuple):
    name: str
    left_api_id: str
    right_api_id: str
    relation: str
    direction: str


CASES = (
    Case(
        "stateful_petstore_order",
        "Advertising/petstore_blitz/getorderbyid",
        "Data/pet_store/getorderbyid",
        "different_capability",
        "none",
    ),
    Case(
        "stateful_petstore_inventory",
        "Business/petstoreapi2_0/getinventory",
        "Other/team_petstore/getinventory",
        "different_capability",
        "none",
    ),
    Case(
        "zip_requires_geocoding",
        "Location/usa_zip_codes_inside_radius/search_by_centre_zip_and_radius",
        "Location/usa_zip_codes_inside_radius/search_by_latitude_longitude_and_radius",
        "different_capability",
        "none",
    ),
    Case(
        "coin_name_requires_registry",
        "Finance/crowdsense/get_all_social_spikes_by_coin_name",
        "Finance/crowdsense/get_all_social_spikes_by_coin_ticker",
        "different_capability",
        "none",
    ),
    Case(
        "book_name_requires_id_mapping",
        "Data/bible_search/get_chapter_by_bookid",
        "Data/bible_search/get_chapter_by_bookname",
        "different_capability",
        "none",
    ),
    Case(
        "state_contains_state_city",
        "Database/restaurants_near_me_usa/get_all_restaurant_locations_by_state_and_city",
        "Database/restaurants_near_me_usa/get_all_restaurants_locations_by_state",
        "contains",
        "right_contains_left",
    ),
    Case(
        "v2_metadata_contains_v1",
        "Social/social_media_data_tt/video_post_metadata",
        "Social/social_media_data_tt/video_post_metadata_v2",
        "contains",
        "right_contains_left",
    ),
    Case(
        "three_month_history_contains_one_month",
        "Finance/stock_prices/get_1_month_historical_daily_prices",
        "Finance/stock_prices/get_3_month_historical_daily_prices",
        "contains",
        "right_contains_left",
    ),
    Case(
        "repeated_rolls_contains_single_roll",
        "Gaming/dice_roll_simulator/regular_dice",
        "Gaming/dice_roll_simulator/regular_dice_rolls",
        "contains",
        "right_contains_left",
    ),
    Case(
        "deterministic_vat_parameter_conversion",
        "Business/eu_vat_number_checker/check_1_param",
        "Business/eu_vat_number_checker/check_2_params",
        "interchangeable",
        "none",
    ),
)


def reverse_direction(direction: str) -> str:
    if direction == "left_contains_right":
        return "right_contains_left"
    if direction == "right_contains_left":
        return "left_contains_right"
    return direction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/retrieval/pooled_pairs.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/retrieval_analysis/v15_regression_pairs.jsonl"),
    )
    args = parser.parse_args()

    requested = {
        frozenset((case.left_api_id, case.right_api_id)): case for case in CASES
    }
    found = {}
    with args.pairs.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            pair = json.loads(line)
            key = frozenset((pair["left_api_id"], pair["right_api_id"]))
            case = requested.get(key)
            if case is None:
                continue
            direction = case.direction
            if pair["left_api_id"] != case.left_api_id:
                direction = reverse_direction(direction)
            pair["regression_expected"] = {
                "case": case.name,
                "relation": case.relation,
                "direction": direction,
            }
            found[key] = pair

    missing = [case.name for key, case in requested.items() if key not in found]
    if missing:
        raise SystemExit("Missing regression pairs: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for case in CASES:
            row = found[frozenset((case.left_api_id, case.right_api_id))]
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Regression pairs: {len(found)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
