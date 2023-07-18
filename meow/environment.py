""" A simulation Environment """

from pydantic import ConfigDict, Field

from .base_model import BaseModel


class Environment(BaseModel):
    """An environment contains all variables that don't depend on the
    geometry/structure itself such as most commonly wavelength and temperature."""

    wl: float = Field(default=1.5, description="the wavelength of the environment")
    T: float = Field(default=25.0, description="the temperature of the environment")
    model_config = ConfigDict(populate_by_name=True, extra="allow")
