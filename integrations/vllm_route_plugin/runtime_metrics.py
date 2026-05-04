"""
Neural-Scalpel vLLM Plugin Runtime Metrics
"""

class RoutePluginMetrics:
    request_count = 0
    swap_count = 0
    rollback_count = 0
    forward_count = 0
    mixed_batch_violation_count = 0
    kv_collision_count = 0
    
    # Track counts per route
    route_counts = {}
    
    # Track route per request_id for recovery from lightweight SchedulerOutput
    request_routes = {}

    # Current active route ID for the upcoming forward pass
    active_route_id = "__base__"

    @classmethod
    def reset(cls):
        cls.request_count = 0
        cls.swap_count = 0
        cls.rollback_count = 0
        cls.forward_count = 0
        cls.mixed_batch_violation_count = 0
        cls.kv_collision_count = 0
        cls.route_counts = {}
        cls.request_routes = {}
        cls.active_route_id = "__base__"

    @classmethod
    def record_request(cls, route_id, request_id=None):
        cls.request_count += 1
        cls.route_counts[route_id] = cls.route_counts.get(route_id, 0) + 1
        if request_id is not None:
            cls.request_routes[str(request_id)] = route_id

    @classmethod
    def get_route_for_request_id(cls, request_id):
        return cls.request_routes.get(str(request_id), "__base__")

    @classmethod
    def record_forward(cls):
        cls.forward_count += 1

    @classmethod
    def record_swap(cls):
        cls.swap_count += 1

    @classmethod
    def record_rollback(cls):
        cls.rollback_count += 1

    @classmethod
    def record_violation(cls):
        cls.mixed_batch_violation_count += 1

    @classmethod
    def set_active_route(cls, route_id):
        cls.active_route_id = route_id or "__base__"

    @classmethod
    def get_active_route(cls):
        return cls.active_route_id
