"""
Representation Learning - Intro (Lecture 1.1)

1. What is representation learning?
   In linear regression, features (like height, weight, age) are given by humans.
   Representation learning is about learning these features automatically from raw data,
   instead of relying on handcrafted ones.
   The learned features are called representations or embeddings.

2. Mapping nodes to d-dimensional embeddings
   In a graph (e.g., social network):
     - Nodes = entities (people)
     - Edges = relationships (friendships)
   We map each node to a vector of length d. For example:
       Alice   -> [0.9, 1.2, -0.3]
       Bob     -> [0.8, 1.0, -0.2]
       Charlie -> [-1.5, 0.4, 2.0]
   These vectors allow us to run standard ML algorithms that expect numeric input.

3. "Similar nodes should be close together"
   The idea is that if two nodes are similar (e.g., have similar neighbors or attributes),
   their vectors should be close in this embedding space.
   Close means small Euclidean distance or high cosine similarity.
   Example:
       Alice and Bob are friends with many of the same people → embeddings close.
       Charlie is from a different community → embedding far apart.

4. Why GNNs care about this
   Graph Neural Networks learn these embeddings automatically by:
       1. Using a node's own features (if available)
       2. Aggregating features from its neighbors
   This process repeats for several layers, making connected and similar nodes
   end up close in embedding space.

5. Relation to linear regression
   In linear regression: feature vector x is given.
   In GNNs: feature vector x is learned from graph structure and optional attributes.
   After learning these vectors, one can use simple models like linear/logistic regression
   for downstream tasks.

One-line intuition:
Representation learning = Let the model figure out the best way to describe each entity
as a vector, so that similar entities have similar coordinates.
"""


"""
Features, Euclidean Distance, and Cosine Similarity

1. Feature = input
   A feature is an input to the model, usually represented as numbers in a vector.
   Example in linear regression:
       Features for house price prediction might be:
       [size_in_sqft, number_of_rooms, distance_to_city]
   In GNNs, after representation learning, each node has a learned feature vector
   (embedding) that can be used as input to downstream models.

2. Euclidean distance
   Measures the straight-line distance between two points in space.
   Formula (for vectors a and b of dimension d):
       Euclidean(a, b) = sqrt( (a1 - b1)^2 + (a2 - b2)^2 + ... + (ad - bd)^2 )
   Example:
       a = [0, 0]
       b = [3, 4]
       distance = sqrt( (3-0)^2 + (4-0)^2 ) = sqrt(9 + 16) = sqrt(25) = 5
   A small Euclidean distance means points are close together.

3. Cosine similarity
   Measures the angle between two vectors, not their distance.
   Formula:
       cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)
       where:
           a · b = dot product of a and b
           ||a|| = length (magnitude) of vector a
           ||b|| = length (magnitude) of vector b
   Range:
       +1   → vectors point in the exact same direction
        0   → vectors are at 90 degrees (no similarity)
       -1   → vectors point in opposite directions
   Example:
       a = [1, 0]
       b = [10, 0]
       cosine similarity = 1 (same direction, different magnitude)

4. Why these are used in GNNs
   After learning embeddings, similarity between two nodes can be measured by:
       - Euclidean distance: how far apart they are in space
       - Cosine similarity: how aligned their direction is in space
"""


"""
Features, Euclidean Distance, and Cosine Similarity

1. Feature = input
   A feature is an input to the model, usually represented as numbers in a vector.
   Example in linear regression:
       Features for house price prediction might be:
       [size_in_sqft, number_of_rooms, distance_to_city]
   In GNNs, after representation learning, each node has a learned feature vector
   (embedding) that can be used as input to downstream models.

2. Euclidean distance
   Measures the straight-line distance between two points in space.
   Formula (for vectors a and b of dimension d):
       Euclidean(a, b) = sqrt( (a1 - b1)^2 + (a2 - b2)^2 + ... + (ad - bd)^2 )
   Example:
       a = [0, 0]
       b = [3, 4]
       distance = sqrt( (3-0)^2 + (4-0)^2 ) = sqrt(9 + 16) = sqrt(25) = 5
   A small Euclidean distance means points are close together.

3. Cosine similarity
   Measures the angle between two vectors, not their distance.
   Formula:
       cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)
       where:
           a · b = dot product of a and b
           ||a|| = length (magnitude) of vector a
           ||b|| = length (magnitude) of vector b
   Range:
       +1   → vectors point in the exact same direction
        0   → vectors are at 90 degrees (no similarity)
       -1   → vectors point in opposite directions
   Example:
       a = [1, 0]
       b = [10, 0]
       cosine similarity = 1 (same direction, different magnitude)

4. Why these are used in GNNs
   After learning embeddings, similarity between two nodes can be measured by:
       - Euclidean distance: how far apart they are in space
       - Cosine similarity: how aligned their direction is in space
"""


"""
Embeddings and Representation Learning

1. Embeddings
   - An embedding is a way of representing an entity (word, image, node in a graph, etc.)
     as a vector of numbers.
   - The vector captures properties of the entity in such a way that similar entities
     have similar vectors.
   - Purpose: turn raw, complex objects into numerical form so they can be processed
     by machine learning models.
   - Example:
       In a social network:
           Alice  -> [0.9, 1.2, -0.3]
           Bob    -> [0.8, 1.0, -0.2]
         Alice and Bob have similar embeddings because they have similar friends.

2. Representation Learning
   - The process of automatically learning these embeddings from data, instead of
     manually designing features.
   - The model learns to map each entity into a vector space where meaningful
     relationships are preserved.
   - In traditional ML:
       Features are handcrafted (decided by humans).
     In representation learning:
       Features are learned by the model during training.

3. Why embeddings are useful
   - They allow complex, unstructured data (text, images, graphs) to be used in
     standard ML pipelines.
   - Similarity between entities can be measured using distance metrics
     (e.g., Euclidean distance, cosine similarity) in the embedding space.

4. In the context of GNNs
   - Each node in the graph gets an embedding.
   - The embedding is learned using both:
       (a) the node's own attributes (if available)
       (b) the attributes of its neighbors and graph structure
   - Nodes that are connected or have similar neighborhoods get embeddings that
     are close together in the learned space.

One-line intuition:
Embeddings are numeric summaries of entities; representation learning is how we
teach a model to create those summaries automatically from raw data.
"""

"""
Representation learning in graphs

- Goal:
    Automatically learn a vector representation (embedding) for each node in a graph
    such that the vector captures important information about the node and its position
    in the graph.

- "Generated by itself" means:
    You do not manually specify the vector values.
    The model learns these values during training by looking at:
        1. The node's own features (if available)
        2. The features of its neighbors
        3. The overall graph structure

- Example:
    Suppose you have a graph of users connected by friendships.
    Initially, you may only know simple info like "age" or "country" for each user.
    A Graph Neural Network will:
        - Pass messages between connected nodes
        - Aggregate neighbor information
        - Update each node's vector step by step
    After training, each node has a learned embedding that reflects both its attributes
    and the context provided by its neighbors.

- Why this is useful:
    The learned embeddings can be fed into other models for:
        - Node classification
        - Link prediction
        - Recommendation systems

One-line intuition:
The model figures out the best way to describe each node as a vector, based on its
own data and its relationships, without you hardcoding those descriptions.
"""



