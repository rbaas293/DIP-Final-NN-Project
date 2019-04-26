# Directory Overview `root/code/_training/imagenet`

* `../` - Location for image sets we download form [imagenet](http://www.image-net.org)

## Why have this directory?

Its purpose is that we can have multiple sets of data(pictures) and whenever we want to run the test
on a diffrent set, all we need to do is copy the contense on `../<set-name>` to `root/code/current_test_data`.