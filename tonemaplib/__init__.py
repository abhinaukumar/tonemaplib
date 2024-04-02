from typing import Type

from .tmo import TMO
from .reinhard02 import Reinhard02TMO
from .boitard12 import Boitard12TMO
from .eilertsen15 import Eilertsen15TMO
from .reinhard12 import Reinhard12TMO
from .itu21 import ITU21TMO
from .hable import HableTMO
from .shan12 import Shan12TMO
from .oskarsson17 import Oskarsson17TMO
from .durand02 import Durand02TMO
from .rana19 import Rana19TMO
from .yang21 import Yang21TMO

_tmo_dict = {
    'Reinhard02': Reinhard02TMO,
    'Boitard12': Boitard12TMO,
    'Eilertsen15': Eilertsen15TMO,
    'Reinhard12': Reinhard12TMO,
    'ITU21': ITU21TMO,
    'Hable': HableTMO,
    'Shan12': Shan12TMO,
    'Oskarsson17': Oskarsson17TMO,
    'Durand02': Durand02TMO,
    'Rana19': Rana19TMO,
    'Yang21': Yang21TMO,
    'Base': TMO
}


def get_tmoclass(name: str) -> Type:
    '''
    Returns TMO class by name. The suffix 'TMO' is not included in the name.

    Args:
        name: Name of the TMO.

    Returns:
        Type: TMO class.

    Raises:
        ValueError: If name is invalid.
    '''
    try:
        return _tmo_dict[name]
    except KeyError:
        raise ValueError('Invalid TMO name')
