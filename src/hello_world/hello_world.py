#!/usr/bin/env python

"""
Hello World class
"""


class HelloWorld:
    """Hello World"""

    __hello_text = "Hello"
    __world_text = "World"

    def hello(self, seperator):
        """
        hello world
        :param seperator: seperator
        :return: Hello World with seperator
        """

        return "Hello" + seperator + self.__world_text

    def world(self, seperator):
        """
        hello world
        :param seperator: seperator
        :return: Hello World with seperator
        """
        return self.__hello_text + seperator + "World"


def main():
    """
    main
    """
    print(HelloWorld().hello(" "))


if __name__ == "__main__":
    main()
