import sqlGenerator as sg

print(sg.generate("Who is the oldest teacher","CREATE TABLE teacher (name VARCHAR, age INTEGER)"))
