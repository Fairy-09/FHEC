# -*- coding: UTF-8 -*-

import pickle
import socket
import time

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

Epoch = 16

def m1(*args):
    import copy
    result = copy.deepcopy(args[0])
    for i in range(1, len(args)):
        for j in range(len(result)):
            result[j] += args[i][j]

    for i in range(len(args)):
        for j in range(len(args[0])):
            args[i][j] = result[j] / len(args)

def m2(*args):
    import copy
    result = copy.deepcopy(args[0])
    for i in range(1, len(args)):
        # print(args[i])
        for j in range(len(result)):
            for k in range(len(result[0])):
                result[j][k] += args[i][j][k]

    for i in range(len(args)):
        for j in range(len(result)):
            for k in range(len(result[0])):
                args[i][j][k] = result[j][k] / len(args)

def socket_udp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '192.168.1.102'
    port = 6138
    s.bind((host, port))
    s.listen(5)
    print('waiting for connecting')

    res, addrs = [], []
    cnt = 1
    while True:
        log("第%d轮开始接收并计时" % cnt)
        try:
            s.settimeout(30000)
            start = time.time()
            # 接收操作
            sock, addr = s.accept()
            print(sock)
            data = sock.recv(1024*1000)
            print('Received from %s:%s' % addr)
            print('Received data:', data)

            tmp = pickle.loads(data)
            print(tmp['num'], cnt)
            if tmp['num'] == cnt:
                addrs.append(sock)
                res.append(tmp['model'])

            recv_time = time.time() - start
            print(len(res))
            if len(res) >= 5 or recv_time > 2000000:
                log("第%d轮接收完毕接收来自%d个节点的参数" % (cnt, len(res)))
                log("开始融合处理操作......")
                # time.sleep(5)
                # res = str(sum(res))
                for m, n in zip(res[0].values(), res[1].values()):
                    if len(m.size()) == 1:
                        m1(m, n)
                    elif len(m.size()) == 2:
                        m2(m, n)
                # print(res[0])
                # res = pickle.dumps(res[0])
                data = {}
                data['num'] = cnt
                data['model'] = res[0]
                log('第%d轮融合完毕，下发......' % cnt)
                data = pickle.dumps(data)
                # print(data)
                for sock in (addrs):
                    sock.send(data)
                    sock.close()
                    # s.sendto(b'%s' % res.encode('utf-8'), addr)
                # else:
                #     res = '处理完毕，关闭连接'
                #     for addr in (addrs):
                #         s.sendto(b'%s' % res.encode('utf-8'), addr)
                #     break
                res, addrs = [], []
                cnt += 1
                if cnt > Epoch:
                    log('处理完毕，关闭连接')
                    break
            else:
                continue
        except:
            log("超时!!!")
            cnt += 1
    s.close()
def main():
    socket_udp_server()

if __name__ == '__main__':
    main()

