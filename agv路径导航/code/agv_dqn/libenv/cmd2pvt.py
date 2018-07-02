
def acce2pvt(is_acce, v0, delta_t = 0.20, acce = 1000.0, dece = 1000.0, maxvel = 2000.0):
    if is_acce > 0:
        tmp = (maxvel - v0) / acce;

        if tmp > delta_t:
            v = v0 + acce*delta_t
            s = (v + v0) / 2 * delta_t
        else:
            v = maxvel
            s = v * (delta_t - tmp) + (maxvel + v0)/2 * tmp

    else:
        tmp = v0 / dece

        if tmp > delta_t:
            v = v0 - dece * delta_t
            s = (v + v0) / 2 * delta_t
        else:
            v = 0
            s = v0 / 2 * tmp

    return s,v,delta_t*1000
