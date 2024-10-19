class NotFound(Exception):
    pass


class ModelNotFound(NotFound):
    pass


class OrganizationNotFound(NotFound):
    pass


class CredentialsNotProvided(Exception):
    pass


class CaptchaDetected(Exception):
    pass


class NotSupported(Exception):
    pass
