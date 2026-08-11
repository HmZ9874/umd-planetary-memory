class UMDAPIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None, details=None):
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details or {}

