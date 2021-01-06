from ._smrMAT import smrMATReader

__all__ = [
			'smrMATReader',
]

readers = [
				(smrMATReader, smrMATReader.name, smrMATReader.standard),
]
