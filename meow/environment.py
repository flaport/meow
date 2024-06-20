""" one place to gather your environment settings """

from __future__ import annotations

from pydantic import ConfigDict, Field

from meow.base_model import BaseModel


class Environment(BaseModel):
    """An environment contains all variables that don't depend on the
    geometry/structure itself such as most commonly wavelength and temperature."""

    wl: float = Field(default=1.5, description="the wavelength of the environment")
    T: float = Field(default=25.0, description="the temperature of the environment")

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )
