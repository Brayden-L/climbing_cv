Various models to accomplish image classification using PyTorch and WandB.

The intent is to allow climbing media aggregators such as [Mountain Project](https://www.mountainproject.com/) or [Openbeta](https://openbeta.io/) group photos similar to how Yelp or Google Maps groups resteraunt photos into menu, dishes, vibe etc.

There is a common perception that so called "butt-shots" are low quality. The reason for this is that the angle blocks compelling areas of the subject and route itself. Because this is the default point of view for a typical belaying partner, these photos are common.

![Butt Shot Example](https://mountainproject.com/assets/photos/climb/106004836_medium_1558390049.jpg?cache=1701315016)

Photos of a higher quality are typically taken from a profile view with a partial landscape in the background, a top down view, or a wide angle head-on shot from a distance back. I've grouped these into a single class referred to as a "glory-shot".

![Glory Shot Example](https://cdn.outsideonline.com/wp-content/uploads/2016/05/19/dawn-wall-free.jpg?width=800)

The final type of typical photo is a topographic photo, which typically does not include a human subject, but focuses on the route itself. There can be drawn overlay to provide guidance of where the route goes. The topo may also be completely hand drawn. These photos are important sources of information, and having them grouped can make information collection more efficient for the user.

![Topo Example]([https://en.wikipedia.org/wiki/Topo_%28climbing%29#/media/File:Routen_Westliche_Zinne_Nord.jpg](https://climbapedia.org/sites/default/files/bixauca%20topo%203.jpg)https://climbapedia.org/sites/default/files/bixauca%20topo%203.jpg)
