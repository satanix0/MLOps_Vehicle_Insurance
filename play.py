from dataclasses import dataclass

@dataclass
class NewClass:
    hel: str


x = NewClass(hel="random")
print(x)