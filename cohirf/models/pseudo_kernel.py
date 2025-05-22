from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

class PseudoKernelClustering:
    def __init__(
            self,
            base_model: str | type[BaseEstimator] = 'kmeans',
            base_model_kwargs: Optional[dict] = None,
            transform_method: str | type[TransformerMixin] = 'kpca',
            transform_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the PseudoKernelClustering class.

        Parameters
        ----------
        base_model : str or type[BaseEstimator], optional
            The base clustering model to use. Default is 'kmeans'.
        base_model_kwargs : dict, optional
            Additional keyword arguments for the base model.
        sampling_method : str or type[TransformerMixin], optional
            The sampling method to use. Default is 'kpca'.
        sampling_kwargs : dict, optional
            Additional keyword arguments for the sampling method.
        """
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs if base_model_kwargs else {}
        self.transform_method = transform_method
        self.transform_kwargs = transform_kwargs if transform_kwargs else {}

    def get_model_instance(self):
        if isinstance(self.base_model, str):
            if self.base_model == 'kmeans':
                model = KMeans(**self.base_model_kwargs)
            else:
                raise ValueError("base_model must be 'kmeans' or a subclass of BaseEstimator")
        elif isinstance(self.base_model, type) and issubclass(self.base_model, BaseEstimator):
            model = self.base_model(**self.base_model_kwargs)
        else:
            raise ValueError("base_model must be a string or a subclass of BaseEstimator")
        return model

    def get_transformer_instance(self):
        if isinstance(self.transform_method, str):
            if self.transform_method == 'kpca':
                transformer = KernelPCA(**self.transform_kwargs)
            else:
                raise ValueError("transform_method must be 'kpca' or a subclass of TransformerMixin")
        elif isinstance(self.transform_method, type) and issubclass(self.transform_method, TransformerMixin):
            transformer = self.transform_method(**self.transform_kwargs)
        else:
            raise ValueError("sampling_method must be a string or a subclass of TransformerMixin")
        return transformer

    def fit(self, X, y=None):
        transformer = self.get_transformer_instance()
        X_transformed = transformer.fit_transform(X)
        self.base_model_instance = self.get_model_instance()
        try:
            self.base_model_instance.fit(X_transformed, y)  # type: ignore
        except AttributeError:
            raise ValueError("The base model must implement the fit method.")
        return self

    def fit_predict(self, X, y=None):
        transformer = self.get_transformer_instance()
        X_transformed = transformer.fit_transform(X)
        self.base_model_instance = self.get_model_instance()
        try:
            return self.base_model_instance.fit_predict(X_transformed, y)  # type: ignore
        except AttributeError:
            raise ValueError("The base model must implement the fit_predict method.")
