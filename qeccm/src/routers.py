import abc


class GenericRouter(metaclass=abc.ABC):
    pass


class SABRERouter(GenericRouter):

    def __init__(self):
        super().__init__()

    def simple_route(self):
        pass

    def _local_routing():
        # Intra
        pass

    def _global_routing():
        # Inter
        pass

    def partitioned_route(self):
        self._global_routing()
        self._local_routing()


    
