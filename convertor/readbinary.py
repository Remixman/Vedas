f = open("/media/noo/hd2/DATA/yago/yago2s-2013-05-08.tid", "rb")
try:
  while True:
    s = f.read(4)
    if s == "":
      break
    p = f.read(4)
    o = f.read(4)
    print s, p, o

    # if p == 168:
      # print s, p, o
finally:
  f.close()