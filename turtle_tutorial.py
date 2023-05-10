import turtle

# Creating turtle screen
t = turtle.Turtle()
# Move turtle in opposite direction
t.backward(100)
# To stop the screen to display
turtle.mainloop()

t.right(90)
t.forward(100)
t.left(90)
t.backward(100)
t.goto(100, 100)

t.home()

t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.clear()

t.circle(50)

t.dot(20)

turtle.bgcolor("red")

turtle.title("asdf")

t.pensize(5)
t.forward(100)

t.shapesize(3, 3, 3)

t.fillcolor("red")

turtle.bgcolor("gray")

t.pencolor("green")

t.color("red", "green")

t.begin_fill()

t.fd(100)
t.lt(120)
t.fd(100)
t.lt(120)
t.fd(100)
t.end_fill()

t.shape("turtle")

t.speed(3)

t.clear()

t.pencolor("purple")
t.fillcolor("orange")
t.pensize(10)
t.speed(9)
t.begin_fill()
t.circle(90)
t.end_fill()

t.clear()

t.pen(pencolor="purple", fillcolor="orange", pensize=10, speed=9)
t.begin_fill()
t.circle(90)
t.end_fill()

t.fd(100)
t.rt(90)
t.penup()
t.fd(100)
t.rt(90)
t.pendown()
t.fd(100)
t.rt(90)
t.penup()
t.fd(100)
t.pendown()

t.reset()

t.stamp()
t.forward(20)

t.clearstamp(24)

c = t.clone()
t.circle(20)
c.circle(30)
c.backward(100)

# for문
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)

t.reset()

for i in range(4):
    t.fd(100)
    t.rt(90)

# while문

n = 10
while n <= 40:
    t.circle(n)
    n = n + 10

# 거북이경주

player_one = turtle.Turtle()
player_one.color("green")
player_one.shape("turtle")
player_one.penup()
player_one.goto(-200, 100)
player_two = player_one.clone()
player_two.color("blue")
player_two.penup()
player_two.goto(-200, -100)

player_one.goto(300, 60)
player_one.pendown()
player_one.circle(40)
player_one.penup()
player_one.goto(-200, 100)
player_two.goto(300, -140)
player_two.pendown()
player_two.circle(40)
player_two.penup()
player_two.goto(-200, -100)

# 거북이경주게임
import random

die = [1, 2, 3, 4, 5, 6]

for i in range(20):
    if player_one.pos() >= (300, 100):
        print("Player One Wins!")
        break
    elif player_two.pos() >= (300, -100):
        print("Player Two Wins!")
        break
    else:
        player_one_turn = input("Press 'Enter' to roll the die ")
        die_outcome = random.choice(die)
        print("The result of the die roll is: ")
        print(die_outcome)
        print("The number of steps will be: ")
        print(20 * die_outcome)
        player_one.fd(20 * die_outcome)
        player_two_turn = input("Press 'Enter' to roll the die ")
        die_outcome = random.choice(die)
        print("The result of the die roll is: ")
        print(die_outcome)
        print("The number of steps will be: ")
        print(20 * die_outcome)
        player_two.fd(20 * die_outcome)

turtle.mainloop()

# tut1

import turtle

# Creating turtle screen
t = turtle.Turtle()

t.heading()
# Move turtle in opposite direction
t.right(25)

t.heading()
# To stop the screen to display
turtle.mainloop()

# 원그리기 응용

import turtle

# Creating turtle
t = turtle.Turtle()

turtle.bgcolor("black")
turtle.pensize(2)
turtle.speed(0)

while (True):
    for i in range(6):
        for colors in ["red", "blue", "magenta", "green", "yellow", "white"]:
            turtle.color(colors)
            turtle.circle(100)
            turtle.left(2)

turtle.hideturtle()
turtle.mainloop()

# 아름다운 예제

import turtle as tu

roo = tu.Turtle()  # Turtle object
wn = tu.Screen()  # Screen Object
wn.bgcolor("black")  # Screen Bg color
wn.title("Fractal Tree Pattern")
roo.left(90)  # moving the turtle 90 degrees towards left
roo.speed(0)  # setting the speed of the turtle


def draw(l):  # recursive function taking length 'l' as argument
    if (l < 10):
        return
    else:

        roo.pensize(2)  # Setting Pensize
        roo.pencolor("yellow")  # Setting Pencolor as yellow
        roo.forward(l)  # moving turtle forward by 'l'
        roo.left(30)  # moving the turtle 30 degrees towards left
        draw(3 * l / 4)  # drawing a fractal on the left of the turtle object 'roo' with 3/4th of its length
        roo.right(60)  # moving the turtle 60 degrees towards right
        draw(3 * l / 4)  # drawing a fractal on the right of the turtle object 'roo' with 3/4th of its length
        roo.left(30)  # moving the turtle 30 degrees towards left
        roo.pensize(2)
        roo.backward(l)  # returning the turtle back to its original psition


draw(20)  # drawing 20 times

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor("magenta")  # magenta
        roo.forward(l)
        roo.left(30)
        draw(3 * l / 4)
        roo.right(60)
        draw(3 * l / 4)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(20)

roo.left(270)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor("red")  # red
        roo.forward(l)
        roo.left(30)
        draw(3 * l / 4)
        roo.right(60)
        draw(3 * l / 4)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(20)

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor('#FFF8DC')  # white
        roo.forward(l)
        roo.left(30)
        draw(3 * l / 4)
        roo.right(60)
        draw(3 * l / 4)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(20)


########################################################

def draw(l):
    if (l < 10):
        return
    else:

        roo.pensize(3)
        roo.pencolor("lightgreen")  # lightgreen
        roo.forward(l)
        roo.left(30)
        draw(4 * l / 5)
        roo.right(60)
        draw(4 * l / 5)
        roo.left(30)
        roo.pensize(3)
        roo.backward(l)


draw(40)

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(3)
        roo.pencolor("red")  # red
        roo.forward(l)
        roo.left(30)
        draw(4 * l / 5)
        roo.right(60)
        draw(4 * l / 5)
        roo.left(30)
        roo.pensize(3)
        roo.backward(l)


draw(40)

roo.left(270)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(3)
        roo.pencolor("yellow")  # yellow
        roo.forward(l)
        roo.left(30)
        draw(4 * l / 5)
        roo.right(60)
        draw(4 * l / 5)
        roo.left(30)
        roo.pensize(3)
        roo.backward(l)


draw(40)

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(3)
        roo.pencolor('#FFF8DC')  # white
        roo.forward(l)
        roo.left(30)
        draw(4 * l / 5)
        roo.right(60)
        draw(4 * l / 5)
        roo.left(30)
        roo.pensize(3)
        roo.backward(l)


draw(40)


########################################################
def draw(l):
    if (l < 10):
        return
    else:

        roo.pensize(2)
        roo.pencolor("cyan")  # cyan
        roo.forward(l)
        roo.left(30)
        draw(6 * l / 7)
        roo.right(60)
        draw(6 * l / 7)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(60)

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor("yellow")  # yellow
        roo.forward(l)
        roo.left(30)
        draw(6 * l / 7)
        roo.right(60)
        draw(6 * l / 7)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(60)

roo.left(270)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor("magenta")  # magenta
        roo.forward(l)
        roo.left(30)
        draw(6 * l / 7)
        roo.right(60)
        draw(6 * l / 7)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(60)

roo.right(90)
roo.speed(2000)


# recursion
def draw(l):
    if (l < 10):
        return
    else:
        roo.pensize(2)
        roo.pencolor('#FFF8DC')  # white
        roo.forward(l)
        roo.left(30)
        draw(6 * l / 7)
        roo.right(60)
        draw(6 * l / 7)
        roo.left(30)
        roo.pensize(2)
        roo.backward(l)


draw(60)
wn.exitonclick()

# 예쁜 꽃무늬

import turtle as tur
import colorsys as cs

# tur.setup(800, 800)
tur.speed(0)
tur.width(3)
tur.bgcolor("black")
for j in range(25):
    for i in range(15):
        tur.color(cs.hsv_to_rgb(i / 15, j / 25, 1))
        tur.right(90)
        tur.circle(200 - j * 4, 90)
        tur.left(90)
        tur.circle(200 - j * 4, 90)
        tur.right(180)
        tur.circle(50, 24)
tur.hideturtle()
tur.done()
