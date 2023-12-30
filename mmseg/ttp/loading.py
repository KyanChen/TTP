from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile(MMCV_LoadImageFromFile):
	"""Load an image pair from files.

	Required Keys:

	- img_path

	Modified Keys:

	- img
	- img_shape
	- ori_shape

	"""

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	def transform(self, results: dict) -> Optional[dict]:
		"""Functions to load image.

		Args:
			results (dict): Result dict from
				:class:`mmengine.dataset.BaseDataset`.

		Returns:
			dict: The dict contains loaded image and meta information.
		"""

		filenames = results['img_path']
		imgs = []
		try:
			for filename in filenames:
				if self.file_client_args is not None:
					file_client = fileio.FileClient.infer_client(
						self.file_client_args, filename)
					img_bytes = file_client.get(filename)
				else:
					img_bytes = fileio.get(
						filename, backend_args=self.backend_args)
				img = mmcv.imfrombytes(
					img_bytes, flag=self.color_type, backend=self.imdecode_backend)
				if self.to_float32:
					img = img.astype(np.float32)
				imgs.append(img)
		except Exception as e:
			if self.ignore_empty:
				return None
			else:
				raise e

		results['img'] = imgs
		results['img_shape'] = imgs[0].shape[:2]
		results['ori_shape'] = imgs[0].shape[:2]
		return results

@TRANSFORMS.register_module()
class LoadMultiImageFromNDArray(MultiImgLoadImageFromFile):

    def transform(self, results: dict) -> dict:

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

