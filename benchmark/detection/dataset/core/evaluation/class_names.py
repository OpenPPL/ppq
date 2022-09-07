# Copyright (c) OpenMMLab. All rights reserved.
import mmcv


def wider_face_classes():
    return ['face']


def voc_classes():
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]


def imagenet_det_classes():
    return [
        'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
        'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
        'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
        'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
        'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
        'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
        'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
        'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
        'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
        'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
        'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
        'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
        'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
        'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
        'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
        'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
        'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
        'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
        'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
        'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
        'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
        'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
        'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
        'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
        'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
        'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
        'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
        'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
        'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
        'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
        'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
        'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
        'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
        'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
        'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
        'whale', 'wine_bottle', 'zebra'
    ]


def imagenet_vid_classes():
    return [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
        'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
        'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
        'watercraft', 'whale', 'zebra'
    ]


def coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]


def cityscapes_classes():
    return [
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]


def oid_challenge_classes():
    return [
        'Footwear', 'Jeans', 'House', 'Tree', 'Woman', 'Man', 'Land vehicle',
        'Person', 'Wheel', 'Bus', 'Human face', 'Bird', 'Dress', 'Girl',
        'Vehicle', 'Building', 'Cat', 'Car', 'Belt', 'Elephant', 'Dessert',
        'Butterfly', 'Train', 'Guitar', 'Poster', 'Book', 'Boy', 'Bee',
        'Flower', 'Window', 'Hat', 'Human head', 'Dog', 'Human arm', 'Drink',
        'Human mouth', 'Human hair', 'Human nose', 'Human hand', 'Table',
        'Marine invertebrates', 'Fish', 'Sculpture', 'Rose', 'Street light',
        'Glasses', 'Fountain', 'Skyscraper', 'Swimwear', 'Brassiere', 'Drum',
        'Duck', 'Countertop', 'Furniture', 'Ball', 'Human leg', 'Boat',
        'Balloon', 'Bicycle helmet', 'Goggles', 'Door', 'Human eye', 'Shirt',
        'Toy', 'Teddy bear', 'Pasta', 'Tomato', 'Human ear',
        'Vehicle registration plate', 'Microphone', 'Musical keyboard',
        'Tower', 'Houseplant', 'Flowerpot', 'Fruit', 'Vegetable',
        'Musical instrument', 'Suit', 'Motorcycle', 'Bagel', 'French fries',
        'Hamburger', 'Chair', 'Salt and pepper shakers', 'Snail', 'Airplane',
        'Horse', 'Laptop', 'Computer keyboard', 'Football helmet', 'Cocktail',
        'Juice', 'Tie', 'Computer monitor', 'Human beard', 'Bottle',
        'Saxophone', 'Lemon', 'Mouse', 'Sock', 'Cowboy hat', 'Sun hat',
        'Football', 'Porch', 'Sunglasses', 'Lobster', 'Crab', 'Picture frame',
        'Van', 'Crocodile', 'Surfboard', 'Shorts', 'Helicopter', 'Helmet',
        'Sports uniform', 'Taxi', 'Swan', 'Goose', 'Coat', 'Jacket', 'Handbag',
        'Flag', 'Skateboard', 'Television', 'Tire', 'Spoon', 'Palm tree',
        'Stairs', 'Salad', 'Castle', 'Oven', 'Microwave oven', 'Wine',
        'Ceiling fan', 'Mechanical fan', 'Cattle', 'Truck', 'Box', 'Ambulance',
        'Desk', 'Wine glass', 'Reptile', 'Tank', 'Traffic light', 'Billboard',
        'Tent', 'Insect', 'Spider', 'Treadmill', 'Cupboard', 'Shelf',
        'Seat belt', 'Human foot', 'Bicycle', 'Bicycle wheel', 'Couch',
        'Bookcase', 'Fedora', 'Backpack', 'Bench', 'Oyster',
        'Moths and butterflies', 'Lavender', 'Waffle', 'Fork', 'Animal',
        'Accordion', 'Mobile phone', 'Plate', 'Coffee cup', 'Saucer',
        'Platter', 'Dagger', 'Knife', 'Bull', 'Tortoise', 'Sea turtle', 'Deer',
        'Weapon', 'Apple', 'Ski', 'Taco', 'Traffic sign', 'Beer', 'Necklace',
        'Sunflower', 'Piano', 'Organ', 'Harpsichord', 'Bed', 'Cabinetry',
        'Nightstand', 'Curtain', 'Chest of drawers', 'Drawer', 'Parrot',
        'Sandal', 'High heels', 'Tableware', 'Cart', 'Mushroom', 'Kite',
        'Missile', 'Seafood', 'Camera', 'Paper towel', 'Toilet paper',
        'Sombrero', 'Radish', 'Lighthouse', 'Segway', 'Pig', 'Watercraft',
        'Golf cart', 'studio couch', 'Dolphin', 'Whale', 'Earrings', 'Otter',
        'Sea lion', 'Whiteboard', 'Monkey', 'Gondola', 'Zebra',
        'Baseball glove', 'Scarf', 'Adhesive tape', 'Trousers', 'Scoreboard',
        'Lily', 'Carnivore', 'Power plugs and sockets', 'Office building',
        'Sandwich', 'Swimming pool', 'Headphones', 'Tin can', 'Crown', 'Doll',
        'Cake', 'Frog', 'Beetle', 'Ant', 'Gas stove', 'Canoe', 'Falcon',
        'Blue jay', 'Egg', 'Fire hydrant', 'Raccoon', 'Muffin', 'Wall clock',
        'Coffee', 'Mug', 'Tea', 'Bear', 'Waste container', 'Home appliance',
        'Candle', 'Lion', 'Mirror', 'Starfish', 'Marine mammal', 'Wheelchair',
        'Umbrella', 'Alpaca', 'Violin', 'Cello', 'Brown bear', 'Canary', 'Bat',
        'Ruler', 'Plastic bag', 'Penguin', 'Watermelon', 'Harbor seal', 'Pen',
        'Pumpkin', 'Harp', 'Kitchen appliance', 'Roller skates', 'Bust',
        'Coffee table', 'Tennis ball', 'Tennis racket', 'Ladder', 'Boot',
        'Bowl', 'Stop sign', 'Volleyball', 'Eagle', 'Paddle', 'Chicken',
        'Skull', 'Lamp', 'Beehive', 'Maple', 'Sink', 'Goldfish', 'Tripod',
        'Coconut', 'Bidet', 'Tap', 'Bathroom cabinet', 'Toilet',
        'Filing cabinet', 'Pretzel', 'Table tennis racket', 'Bronze sculpture',
        'Rocket', 'Mouse', 'Hamster', 'Lizard', 'Lifejacket', 'Goat',
        'Washing machine', 'Trumpet', 'Horn', 'Trombone', 'Sheep',
        'Tablet computer', 'Pillow', 'Kitchen & dining room table',
        'Parachute', 'Raven', 'Glove', 'Loveseat', 'Christmas tree',
        'Shellfish', 'Rifle', 'Shotgun', 'Sushi', 'Sparrow', 'Bread',
        'Toaster', 'Watch', 'Asparagus', 'Artichoke', 'Suitcase', 'Antelope',
        'Broccoli', 'Ice cream', 'Racket', 'Banana', 'Cookie', 'Cucumber',
        'Dragonfly', 'Lynx', 'Caterpillar', 'Light bulb', 'Office supplies',
        'Miniskirt', 'Skirt', 'Fireplace', 'Potato', 'Light switch',
        'Croissant', 'Cabbage', 'Ladybug', 'Handgun', 'Luggage and bags',
        'Window blind', 'Snowboard', 'Baseball bat', 'Digital clock',
        'Serving tray', 'Infant bed', 'Sofa bed', 'Guacamole', 'Fox', 'Pizza',
        'Snowplow', 'Jet ski', 'Refrigerator', 'Lantern', 'Convenience store',
        'Sword', 'Rugby ball', 'Owl', 'Ostrich', 'Pancake', 'Strawberry',
        'Carrot', 'Tart', 'Dice', 'Turkey', 'Rabbit', 'Invertebrate', 'Vase',
        'Stool', 'Swim cap', 'Shower', 'Clock', 'Jellyfish', 'Aircraft',
        'Chopsticks', 'Orange', 'Snake', 'Sewing machine', 'Kangaroo', 'Mixer',
        'Food processor', 'Shrimp', 'Towel', 'Porcupine', 'Jaguar', 'Cannon',
        'Limousine', 'Mule', 'Squirrel', 'Kitchen knife', 'Tiara', 'Tiger',
        'Bow and arrow', 'Candy', 'Rhinoceros', 'Shark', 'Cricket ball',
        'Doughnut', 'Plumbing fixture', 'Camel', 'Polar bear', 'Coin',
        'Printer', 'Blender', 'Giraffe', 'Billiard table', 'Kettle',
        'Dinosaur', 'Pineapple', 'Zucchini', 'Jug', 'Barge', 'Teapot',
        'Golf ball', 'Binoculars', 'Scissors', 'Hot dog', 'Door handle',
        'Seahorse', 'Bathtub', 'Leopard', 'Centipede', 'Grapefruit', 'Snowman',
        'Cheetah', 'Alarm clock', 'Grape', 'Wrench', 'Wok', 'Bell pepper',
        'Cake stand', 'Barrel', 'Woodpecker', 'Flute', 'Corded phone',
        'Willow', 'Punching bag', 'Pomegranate', 'Telephone', 'Pear',
        'Common fig', 'Bench', 'Wood-burning stove', 'Burrito', 'Nail',
        'Turtle', 'Submarine sandwich', 'Drinking straw', 'Peach', 'Popcorn',
        'Frying pan', 'Picnic basket', 'Honeycomb', 'Envelope', 'Mango',
        'Cutting board', 'Pitcher', 'Stationary bicycle', 'Dumbbell',
        'Personal care', 'Dog bed', 'Snowmobile', 'Oboe', 'Briefcase',
        'Squash', 'Tick', 'Slow cooker', 'Coffeemaker', 'Measuring cup',
        'Crutch', 'Stretcher', 'Screwdriver', 'Flashlight', 'Spatula',
        'Pressure cooker', 'Ring binder', 'Beaker', 'Torch', 'Winter melon'
    ]


def oid_v6_classes():
    return [
        'Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football',
        'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy',
        'Organ (Musical Instrument)', 'Cassette deck', 'Apple', 'Human eye',
        'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard',
        'Bird', 'Parking meter', 'Traffic light', 'Croissant', 'Cucumber',
        'Radish', 'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick',
        'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle',
        'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot',
        'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy',
        'Screwdriver', 'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt',
        'Drill (Tool)', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Brown bear',
        'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot',
        'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee',
        'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw',
        'Balloon', 'Wrench', 'Tent', 'Vehicle registration plate', 'Lantern',
        'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace',
        'Carnivore', 'Scissors', 'Stairs', 'Computer keyboard', 'Printer',
        'Traffic sign', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock',
        'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 'Watercraft',
        'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile',
        'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel',
        'Coat', 'Suit', 'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola',
        'Beetle', 'Cannon', 'Computer mouse', 'Cookie', 'Office building',
        'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor',
        'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment',
        'Studio couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini',
        'Ladle', 'Human mouth', 'Dairy Product', 'Dice', 'Oven', 'Dinosaur',
        'Ratchet (Device)', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula',
        'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser',
        'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero',
        'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Can opener',
        'Goggles', 'Human body', 'Roller skates', 'Coffee cup',
        'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign',
        'Office supplies', 'Volleyball (Ball)', 'Vase', 'Slow cooker',
        'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 'Personal care', 'Food',
        'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove',
        'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax',
        'Fruit', 'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart',
        'Treadmill', 'Fox', 'Flag', 'French horn', 'Window blind',
        'Human foot', 'Golf cart', 'Jacket', 'Egg (Food)', 'Street light',
        'Guitar', 'Pillow', 'Human leg', 'Isopod', 'Grape', 'Human ear',
        'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle',
        'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat',
        'Baseball bat', 'Baseball glove', 'Mixing bowl',
        'Marine invertebrates', 'Kitchen utensil', 'Light switch', 'House',
        'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed',
        'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer',
        'Harpsichord', 'Human hair', 'Heater', 'Harmonica', 'Hamster',
        'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw',
        'Insect', 'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate',
        'Food processor', 'Bookcase', 'Refrigerator', 'Wood-burning stove',
        'Punching bag', 'Common fig', 'Cocktail shaker', 'Jaguar (Animal)',
        'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet',
        'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife',
        'Bottle', 'Bottle opener', 'Lynx', 'Lavender (Plant)', 'Lighthouse',
        'Dumbbell', 'Human head', 'Bowl', 'Humidifier', 'Porch', 'Lizard',
        'Billiard table', 'Mammal', 'Mouse', 'Motorcycle',
        'Musical instrument', 'Swim cap', 'Frying pan', 'Snowplow',
        'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 'Milk',
        'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom',
        'Crutch', 'Pitcher (Container)', 'Mirror', 'Personal flotation device',
        'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard',
        'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball',
        'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl',
        'Plant', 'Potato', 'Hair spray', 'Sports equipment', 'Pasta',
        'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 'Mixer',
        'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile',
        'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda',
        'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood',
        'Submarine sandwich', 'Snowboard', 'Sword', 'Picture frame', 'Sushi',
        'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine',
        'Scorpion', 'Segway', 'Training bench', 'Snake', 'Coffee table',
        'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco',
        'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree',
        'Tomato', 'Train', 'Tool', 'Picnic basket', 'Cooking spray',
        'Trousers', 'Bowling equipment', 'Football helmet', 'Truck',
        'Measuring cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag',
        'Paper cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale',
        'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 'Lion',
        'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck',
        'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper',
        'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog',
        'Banana', 'Rocket', 'Wine glass', 'Countertop', 'Tablet computer',
        'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark',
        'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser',
        'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree', 'Hamburger',
        'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus',
        'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull',
        'Oyster', 'Horizontal bar', 'Convenience store', 'Bomb', 'Bench',
        'Ice cream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange',
        'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet',
        'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut',
        'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera',
        'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable',
        'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish',
        'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple',
        'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Common sunflower',
        'Microwave oven', 'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug',
        'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow',
        'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone',
        'Sports uniform', 'Tennis racket', 'Wall clock', 'Serving tray',
        'Kitchen & dining room table', 'Dog bed', 'Cake stand',
        'Cat furniture', 'Bathroom accessory', 'Facial tissue holder',
        'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler',
        'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry',
        'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 'Turkey', 'Lily',
        'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant',
        'Car', 'Aircraft', 'Human hand', 'Skunk', 'Teddy bear', 'Watermelon',
        'Cantaloupe', 'Dishwasher', 'Flute', 'Balance beam', 'Sandwich',
        'Shrimp', 'Sewing machine', 'Binoculars', 'Rays and skates', 'Ipod',
        'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume',
        'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair',
        'Rugby ball', 'Armadillo', 'Maracas', 'Helmet'
    ]


dataset_aliases = {
    'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
    'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
    'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
    'coco': ['coco', 'mscoco', 'ms_coco'],
    'wider_face': ['WIDERFaceDataset', 'wider_face', 'WIDERFace'],
    'cityscapes': ['cityscapes'],
    'oid_challenge': ['oid_challenge', 'openimages_challenge'],
    'oid_v6': ['oid_v6', 'openimages_v6']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
