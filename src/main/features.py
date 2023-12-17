from abc import ABC, abstractmethod

from kymatio.torch import Scattering1D, TimeFrequencyScattering


class AcousticFeature(ABC):
    def __init__(self, sr=16000, batch_size=1) -> None:
        super().__init__()
        self.sr = sr
        self.batch_size = batch_size

    def to_device(self, device='cpu'):
        self.transform = (
            self.transform.cuda() if device == "cuda" else self.transform.cpu()
        )

    @abstractmethod
    def compute_features(self):
        raise NotImplementedError("This method must implement computation of" +
                                  " the features (dependent on subclass).")


class MFCC(AcousticFeature):
    pass


class Scat1D(AcousticFeature):
    """
    Computes the 1D scattering transform of an audio signal.

    Attributes
    ----------
    shape : int
        The length of each audio signal (# of samples).
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    device : str
        The device to use for computation (either 'cpu' or 'cuda').
    J : int
        The maximum log-scale of the scattering transform
    Q : int or tuple 
        The number of wavelets for the first and second orders of the
        scattering transform (Q1, Q2). If Q is an int, this corresponds to
        choosing (Q1=Q, Q2=1).
    T : int
        The temporal support of the low-pass filter, controlling amount of
        imposed time-shift invariance. If None, T=2**J.
    global_avg : bool
        Whether or not to compute the global average of each scattering 
        coefficient when computing features.

    """
    def __init__(self,
                 shape, 
                 sr=16000, 
                 batch_size=1,
                 device='cpu',
                 J=8,
                 Q=(12, 2),
                 T=None,
                 global_avg=True) -> None:
        super().__init__(sr, batch_size)
        self.transform = Scattering1D(shape=shape, J=J, Q=Q, T=T)
        self.to_device(device)
        self.global_avg = global_avg
    
    def compute_features(self, X):
        """
        Computes the 1D scattering transform for the given audio signal(s).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, shape), which are the audio signal(s)
            to compute the 1D scattering transform for.
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, C), where C is the number of
            scattering coefficients (if global_avg is True).

        """
        fts = self.transform(X)
        if self.global_avg:
            fts = fts.mean(dim=-1)
        return fts


class JTFS(AcousticFeature):
    """
    Computes the joint-time frequency scattering transform of an audio signal.

    Attributes
    ----------
    shape : int
        The length of each audio signal (# of samples).
    sr : int
        The sample rate of each audio signal.
    batch_size : int
        The number of audio signals to process at once.
    device : str
        The device to use for computation (either 'cpu' or 'cuda').
    J : int
        The maximum log-scale of the scattering transform applied across the
        time axis.
    Q : int or tuple 
        The number of wavelets for the first and second orders of the
        scattering transform (Q1, Q2) applied across the time axis. If Q is an 
        int, this corresponds to choosing (Q1=Q, Q2=1).
    T : int
        The support of the low-pass filter applied across the time axis, 
        controlling amount of imposed time-shift invariance. If None, T=2**J.
    J_fr : int
        The maximum log-scale of the scattering transform applied across the
        frequency axis.
    Q_fr : int
        The number of wavelets of the scattering transform applied across
        the frequency axis.
    F : int
        The support of the low-pass filter applied across the frequency axis,
        controlling amount of imposed frequency-shift invariance. If None,
        F=2**J_fr.
    global_avg : bool
        Whether or not to compute the global average of each scattering 
        coefficient when computing features.
    """
    def __init__(self, 
                 shape,
                 sr=16000, 
                 batch_size=1,
                 device='cpu',
                 J=8,
                 Q=(12, 2),
                 T=None,
                 J_fr=3,
                 Q_fr=2,
                 F=None,
                 global_avg=True) -> None:
        super().__init__(sr, batch_size)
        self.transform = TimeFrequencyScattering(shape=shape,
                                                 J=J,
                                                 Q=Q,
                                                 T=T,
                                                 J_fr=J_fr,
                                                 Q_fr=Q_fr,
                                                 F=F)
        self.to_device(device)
        self.global_avg = global_avg

    def compute_features(self, X):
        """
        Computes the joint time-frequency scattering transform for the given 
        audio signal(s).

        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, shape), which are the audio signal(s)
            to compute the 1D scattering transform for.
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, C), where C is the number of
            scattering coefficients (if global_avg is True).
        
        """
        fts = self.transform(X)
        if self.global_avg:
            fts = fts.mean(dim=-1)
        return fts
