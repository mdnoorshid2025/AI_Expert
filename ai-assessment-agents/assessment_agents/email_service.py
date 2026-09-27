import smtplib
import ssl

from email.message import EmailMessage

from .config import (
    SEND_EMAIL,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM
)

from .state import AssessmentState


def send_email(
    to_address: str,
    subject: str,
    body: str
) -> str:

    # During development we can disable actual sending.
    if not SEND_EMAIL:
        return (
            "Email sending is disabled. "
            "The report was generated successfully."
        )

    # Validate SMTP configuration before connecting.
    if not SMTP_HOST:
        return "Email failed: SMTP_HOST is missing."

    if not SMTP_USER:
        return "Email failed: SMTP_USER is missing."

    if not SMTP_PASSWORD:
        return "Email failed: SMTP_PASSWORD is missing."

    if not SMTP_FROM:
        return "Email failed: SMTP_FROM is missing."

    # Build the email message.
    message = EmailMessage()

    message["From"] = SMTP_FROM
    message["To"] = to_address
    message["Subject"] = subject

    message.set_content(body)

    # Create secure TLS configuration.
    ssl_context = ssl.create_default_context()

    try:

        with smtplib.SMTP(
            SMTP_HOST,
            SMTP_PORT,
            timeout=30
        ) as smtp:

            # Identify ourselves to SMTP server.
            smtp.ehlo()

            # Upgrade connection to TLS.
            smtp.starttls(
                context=ssl_context
            )

            smtp.ehlo()

            # Authenticate sender.
            smtp.login(
                SMTP_USER,
                SMTP_PASSWORD
            )

            # Send email.
            smtp.send_message(
                message
            )

        return (
            f"Email sent successfully "
            f"to {to_address}"
        )

    except smtplib.SMTPAuthenticationError:

        return (
            "Email failed: SMTP authentication failed. "
            "Check SMTP username/app password."
        )

    except smtplib.SMTPException as exc:

        return (
            f"Email failed due to SMTP error: {exc}"
        )

    except OSError as exc:

        return (
            f"Email failed due to network error: {exc}"
        )


def send_email_node(
    state: AssessmentState
):

    recipient = state[
        "raw_requirements"
    ]["email"]

    status = send_email(
        to_address=recipient,
        subject=state[
            "email_subject"
        ],
        body=state[
            "email_body"
        ]
    )

    return {
        "email_status": status
    }