ACTION_DIM = 40
def acce2pvt(pi_local, cur_x, cur_y, cur_theta):
    if pi_local<=(ACTION_DIM/2-1)  and pi_local>=0:
        if abs(cur_theta) <2 or abs(cur_theta - 180) <2 or abs(cur_theta + 180) <2:
            next_x = 500*pi_local+1750
            return next_x,cur_theta
        if abs(cur_theta - 90) < 2 or abs(cur_theta + 90) < 2:
            next_x = 500 * pi_local + 1750
            next_theta = cur_theta + 90
            if abs(next_theta - 90) < 2:
                next_theta = 90
            if abs(next_theta - 180) < 2:
                next_theta = 180
            if abs(next_theta + 90) < 2:
                next_theta = -90
            if abs(next_theta) < 2:
                next_theta = 0
            if abs(next_theta + 180) < 2:
                next_theta = 180
            return next_x,next_theta
    if pi_local <= (ACTION_DIM-1) and pi_local >= (ACTION_DIM/2):
        if abs(cur_theta) < 1 or abs(cur_theta - 180) < 1 or abs(cur_theta + 180) < 1:
            next_y = 500 * (pi_local-ACTION_DIM/2) + 1750
            next_theta = cur_theta + 90
            if abs(next_theta - 90) < 2:
                next_theta = 90
            if abs(next_theta - 180) < 2:
                next_theta = 180
            if abs(next_theta + 90) < 2:
                next_theta = -90
            if abs(next_theta) < 2:
                next_theta = 0
            if abs(next_theta + 180) < 2:
                next_theta = 180
            return next_y, next_theta
        if abs(cur_theta - 90) < 2 or abs(cur_theta + 90) < 2:
            next_y = 500 * (pi_local-ACTION_DIM/2) + 1750
            return next_y, cur_theta


