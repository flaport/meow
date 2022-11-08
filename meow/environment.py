""" A simulation Environment """

from pydantic import Extra

from .base_model import BaseModel


class Environment(BaseModel):
    """An environment contains all variables that don't depend on the
        geometry/structure itself such as most commonly wavelength and temperature.

    Attributes:
        wl: the wavelength of the environment
        T: the temperature of the environment
    """

    wl: float = 1.5
    T: float = 25.0

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow
