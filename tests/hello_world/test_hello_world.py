"""
Hello World Tests
"""

from hello_world import HelloWorld


def test_world():
    # given
    hw = HelloWorld()

    # when
    res = hw.world(" ")

    # then
    assert res == "Hello World"


def test_hello():
    # given
    hw = HelloWorld()

    # when
    res = hw.hello(" ")

    # then
    assert res == "Hello World"
