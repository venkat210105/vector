"""Locust scenario: simulated users hitting the vector-db REST API.

Locust runs many "users" at once. Each user is a class, and each method
decorated with @task is something that user does over and over, picking
tasks at random according to their weight. `wait_time` is the pause a user
takes between actions, like a real person (or client) would.

Run it through `run_load_test.py`, which starts a throwaway server and
sweeps the user count. To run Locust by hand against a server you started
yourself:

    locust -f benchmarks/load_test/locustfile.py --host http://127.0.0.1:8001
"""

import os
import random
import uuid

from locust import HttpUser, between, task

COLLECTION = os.environ.get("LOAD_COLLECTION", "load")
DIM = int(os.environ.get("LOAD_DIM", "16"))


def random_vector() -> list[float]:
    return [random.gauss(0, 1) for _ in range(DIM)]


class SearchUser(HttpUser):
    """A client that only reads. `weight` sets how common this kind of user is
    relative to the other classes: 4 searchers for every 1 writer."""

    weight = 4
    wait_time = between(0.05, 0.2)

    @task
    def search(self):
        # `name=` groups every request under one label in the results table.
        self.client.post(
            f"/collections/{COLLECTION}/search",
            json={"vector": random_vector(), "k": 10, "ef_search": 50},
            name="search",
        )


class WriteUser(HttpUser):
    """A client that only writes. Every write is a new unique id, so the index
    keeps growing during the test, as it would in real use."""

    weight = 1
    wait_time = between(0.05, 0.2)

    @task
    def upsert(self):
        self.client.post(
            f"/collections/{COLLECTION}/vectors",
            json={"id": f"w-{uuid.uuid4().hex}", "vector": random_vector(), "metadata": {}},
            name="upsert",
        )
