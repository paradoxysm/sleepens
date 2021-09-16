default_map = { 'AW': 0, 'QW': 1, 'NR': 2, 'R': 3 }

from ._min_rem import MinREM
from ._rem_drop import REMDrop
from ._rem_sensitivity import REMSensitivity
from ._transition_fix import TransitionFix
from ._wake_max import WakeMax
from ._wake_sensitivity import WakeSensitivity
from ._wake_to_rem import WakeToREM

__all__ = [
			'MinRem',
			'RemDrop',
			'REMSensitivity',
			'TransitionFix',
			'WakeMax',
			'WakeSensitivity',
			'WakeToREM',
		]
