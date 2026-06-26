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
    def filter_schema(cls, required_fields: list[str]):
        original_top_fields = cls.model_fields
        original_sub_fields = VRDULineItem.model_fields

        field_definitions = {}

        line_item_targets = [f for f in required_fields if f in original_sub_fields]

        if line_item_targets:
            sub_field_defs = {}
            for sub_field in line_item_targets:
                field_info = original_sub_fields[sub_field]
                sub_field_defs[sub_field] = (field_info.annotation, field_info.default)

            DynamicLineItem = create_model("DynamicLineItem", **sub_field_defs)
            field_definitions["line_items"] = (
                Optional[List[DynamicLineItem]],
                Field(
                    description="List of all tabular rows. You must populate this array."
                ),
            )

        for field_name in required_fields:
            if field_name in original_top_fields and field_name != "line_items":
                field_info = original_top_fields[field_name]
                field_definitions[field_name] = (
                    field_info.annotation,
                    field_info.default,
                )
                logger.debug(f"Field '{field_name}' included in dynamic schema.")
            elif field_name not in original_sub_fields:
                logger.warning(f"Field '{field_name}' not found. Skipping.")

        dynamic_name = f"Dynamic{cls.__name__}_{abs(hash(tuple(required_fields)))}"
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
