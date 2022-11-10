""" A simulation Environment """

from pydantic import Extra, Field

from .base_model import BaseModel


class Environment(BaseModel):
    """An environment contains all variables that don't depend on the
    geometry/structure itself such as most commonly wavelength and temperature."""

    wl: float = Field(default=1.5, description="the wavelength of the environment")
    T: float = Field(default=25.0, description="the temperature of the environment")

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow
