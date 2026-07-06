from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

import logging

logger = logging.getLogger(__name__)


class VRDULineItem(BaseModel):
    """Sub-Schema for line items (matches the 'line_item' pattern in DeepForm)."""

    channel: Optional[str] = Field(
        None, description="Broadcasting channel (e.g., KMEG)"
    )
    program_start_date: Optional[str] = Field(
        None, description="Start date of the program in MM/DD/YY format"
    )
    program_end_date: Optional[str] = Field(
        None, description="End date of the program in MM/DD/YY format"
    )
    program_desc: Optional[str] = Field(
        None, description="Name or description of the program (e.g., Judge Judy)"
    )
    sub_amount: Optional[str] = Field(
        None, description="Amount of the individual line item (PriceMatch)"
    )


class VRDUBaseSchema(BaseModel):
    """
    Main schema for VRDU documents (Ad-Buy Forms).
    Contains all potential fields from the DeepForm dataset with appropriate metadata.
    """

    property: Optional[str] = Field(
        None, description="Name of the Property/Station (GeneralStringMatch)"
    )
    contract_num: Optional[str] = Field(
        None, description="Contract Number / Order Number (NumericalStringMatch)"
    )
    tv_address: Optional[str] = Field(
        None, description="Address of the TV Station (AddressMatch)"
    )
    flight_to: Optional[str] = Field(
        None, description="End date of the campaign (Flight Dates) - DateMatch format"
    )
    flight_from: Optional[str] = Field(
        None, description="Start date of the campaign (Flight Dates) - DateMatch format"
    )
    advertiser: Optional[str] = Field(
        None, description="Name of the Advertiser (GeneralStringMatch)"
    )
    agency: Optional[str] = Field(
        None, description="Name of the Advertising Agency (GeneralStringMatch)"
    )
    product: Optional[str] = Field(
        None, description="Product number or Product Description (GeneralStringMatch)"
    )
    gross_amount: Optional[str] = Field(
        None, description="Gross Total Amount (PriceMatch)"
    )

    # Tabular Data
    line_items: Optional[List[VRDULineItem]] = Field(
        default_factory=list,
        description="List of all individual positions (Spots/Programs)",
    )

    @classmethod
    def filter_schema(cls, required_fields):
        """
        required_fields: Dict from JSONL,
        z.B. {"property": "string", ..., "line_items": [{"channel": "string", ...}]}
        """
        original_top_fields = cls.model_fields
        original_sub_fields = VRDULineItem.model_fields
        field_definitions = {}

        line_item_subfields = None
        top_level_names = []

        # Sub-Felder für line_items aus der verschachtelten Struktur lesen
        if isinstance(required_fields, dict):
            li_spec = required_fields.get("line_items")
            if isinstance(li_spec, list) and li_spec and isinstance(li_spec[0], dict):
                line_item_subfields = list(li_spec[0].keys())
            top_level_names = list(required_fields.keys())
        else:
            # Fallback, falls doch eine Liste reinkommt
            top_level_names = list(required_fields)
            line_item_subfields = [
                f for f in top_level_names if f in original_sub_fields
            ]

        if line_item_subfields:
            sub_field_defs = {}
            for sub_field in line_item_subfields:
                if sub_field in original_sub_fields:
                    field_info = original_sub_fields[sub_field]
                    sub_field_defs[sub_field] = (
                        field_info.annotation,
                        field_info.default,
                    )

            if sub_field_defs:
                DynamicLineItem = create_model("DynamicLineItem", **sub_field_defs)
                field_definitions["line_items"] = (
                    Optional[List[DynamicLineItem]],
                    Field(
                        default_factory=list, description="List of all tabular rows."
                    ),
                )

        for field_name in top_level_names:
            if field_name == "line_items":
                continue
            if field_name in original_top_fields:
                field_info = original_top_fields[field_name]
                field_definitions[field_name] = (
                    field_info.annotation,
                    field_info.default,
                )

        dynamic_name = f"Dynamic_{cls.__name__}_{abs(hash(str(top_level_names)))}"
        return create_model(dynamic_name, __base__=BaseModel, **field_definitions)


class DocileBaseSchema(BaseModel):
    """
    Base Schema for Docile dataset.
    Adapts dynamically to the fields in your JSON corpus.
    """

    id: Optional[str] = Field(None, description="Document ID")

    @classmethod
    def filter_schema(cls, required_fields: list[str]) -> Type["DocileBaseSchema"]:
        """Filtering logic analogous to VRDUBaseSchema for Docile datasets."""
        return VRDUBaseSchema.filter_schema(required_fields)
