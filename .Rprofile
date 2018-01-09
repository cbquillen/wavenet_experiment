p <- function(x, ...) {
    plot(tail(as.data.frame(x), -1000), type='l', log='xy', ...)
}

ldat <- function(f) {
    return (lowess(scan(f),f=.1))
}

comp <- function(o1='o1', o2='o2') {
    x1 = tail(as.data.frame(ldat(o1)), -1000)
    x2 = tail(as.data.frame(ldat(o2)), -1000)
    plot(x1, xlim=range(x1$x, x2$x), ylim=range(x1$y, x2$y), type='l', log='xy')
    lines(x2, col='red')
}

#c2 <- function() {
#    x1 = tail(as.data.frame(ldat('o1')), -1000)
#    x2 = tail(as.data.frame(ldat('o2')), -1000)
#    library(ggplot2)
#    var='x1'
#    ggplot() + geom_line(aes(x1$x, x1$y, col=var)) + geom_line(aes(x2$x, x2$y, col='x2')) + scale_x_log10() + scale_y_log10()
#}

lv <- function(file, n=1000) {
    return(sqrt(var(tail(scan(file), n))))
}


