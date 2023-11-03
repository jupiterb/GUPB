from typing import TypeVar, Generic
import threading


T = TypeVar("T")


class Observer(Generic[T]):
    def __init__(self) -> None:
        self.__state: T | None = None
        self.__cv = threading.Condition()
        self.__stop = False

    def update(self, state: T) -> None:
        with self.__cv:
            self.__state = state
            self.__cv.notify_all()

    def stop_waiting(self) -> None:
        with self.__cv:
            self.__stop = True
            self.__cv.notify_all()

    def wait_for_observed(self) -> T:
        self.__stop = False
        with self.__cv:
            while self.__state is None and not self.__stop:
                self.__cv.wait()
            state = self.__state
            self.__state = None

        return state


class Observable(Generic[T]):
    def __init__(self) -> None:
        self._observers: list[Observer[T]] = []
        self.__state: T | None = None

    def attach(self, observer: Observer[T]) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer[T]) -> None:
        self._observers.remove(observer)

    @property
    def observable_state(self) -> T | None:
        return self.__state

    @observable_state.setter
    def observable_state(self, state: T) -> None:
        self.__state = state
        self._notify()

    def _notify(self) -> None:
        if self.__state is not None:
            for observer in self._observers:
                observer.update(self.__state)
