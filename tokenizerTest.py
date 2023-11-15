from data.tokenizer import TinyTokenizer

if __name__ == "__main__":
    tknz = TinyTokenizer("train")

    print("tknz.vocab_size()", tknz.vocab_size())
    print("tknz.sp.bos_id()", tknz.sp.bos_id())
    print("tknz.sp.pad_id()", tknz.sp.pad_id())
    print("tknz.sp.eos_id()", tknz.sp.eos_id())
    print("tknz.sp.unk_id()", tknz.sp.unk_id())

    ids_foo = tknz.encode("hello my name is Bes")
    ids_bar = tknz.encode("ciao il mio nome è Bes")
    ids_zoo = tknz.encode("emma")
    print("ids_foo", ids_foo)
    print("ids_bar", ids_bar)
    print("ids_zoo", ids_zoo)
    txt_foo = tknz.decode(ids_foo)
    txt_bar = tknz.decode(ids_bar)
    txt_zoo = tknz.decode(ids_zoo)
    print("txt_foo", txt_foo)
    print("txt_bar", txt_bar)
    print("txt_zoo", txt_zoo)
    for id in range(4):
        print(id, tknz.sp.id_to_piece(id), tknz.sp.is_control(id))
