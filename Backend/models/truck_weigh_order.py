from datetime import datetime

from . import db

OPEN_STATUSES = frozenset({"awaiting_first", "awaiting_second"})


class TruckWeighOrder(db.Model):
    __tablename__ = "truck_weigh_orders"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    ticket = db.Column(db.String(32), unique=True, nullable=False)
    truck_id = db.Column(db.Integer, nullable=False, index=True)
    material_code = db.Column(db.String(50), nullable=False)
    material_name = db.Column(db.String(200))
    first_weight_kg = db.Column(db.Float)
    first_ts = db.Column(db.DateTime(timezone=True))
    second_weight_kg = db.Column(db.Float)
    second_ts = db.Column(db.DateTime(timezone=True))
    net_kg = db.Column(db.Float)
    status = db.Column(db.String(20), nullable=False, default="awaiting_first", index=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), index=True)
    # Silo stock outcome. stock_delta_kg is signed: truck in heavy / out light => positive.
    # stock_applied_at is set as soon as the database quantity has been moved (even
    # while stock_status is still 'pending'), and is what stops a retry from adjusting
    # the database twice.
    silo_no = db.Column(db.Integer)
    stock_delta_kg = db.Column(db.Float)
    stock_status = db.Column(db.String(20))
    stock_qty_before = db.Column(db.Float)
    stock_qty_after = db.Column(db.Float)
    stock_applied_at = db.Column(db.DateTime(timezone=True))
    stock_note = db.Column(db.String(255))

    def compute_net(self):
        if self.first_weight_kg is None or self.second_weight_kg is None:
            return None
        return abs(float(self.second_weight_kg) - float(self.first_weight_kg))

    def site_status(self):
        if self.status == "awaiting_second":
            return "out_pending"
        if self.status == "awaiting_first":
            return "awaiting_first_weight"
        if self.status == "completed":
            return "completed"
        return self.status

    def to_dict(self, truck_plate=None, truck_driver=None):
        return {
            "id": self.id,
            "ticket": self.ticket,
            "truck_id": self.truck_id,
            "truck_plate": truck_plate,
            "truck_driver": truck_driver,
            "material_code": self.material_code,
            "material_name": self.material_name,
            "first_weight_kg": self.first_weight_kg,
            "first_ts": self.first_ts.isoformat() if self.first_ts else None,
            "second_weight_kg": self.second_weight_kg,
            "second_ts": self.second_ts.isoformat() if self.second_ts else None,
            "net_kg": self.net_kg,
            "status": self.status,
            "site_status": self.site_status(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "silo_no": self.silo_no,
            "stock_delta_kg": self.stock_delta_kg,
            "stock_status": self.stock_status,
            "stock_qty_before": self.stock_qty_before,
            "stock_qty_after": self.stock_qty_after,
            "stock_applied_at": self.stock_applied_at.isoformat() if self.stock_applied_at else None,
            "stock_note": self.stock_note,
        }


def ensure_truck_weigh_orders_table():
    """Create truck_weigh_orders on existing installs (idempotent)."""
    from sqlalchemy import inspect, text

    db.create_all()
    insp = inspect(db.engine)
    if not insp.has_table("truck_weigh_orders"):
        return

    index_statements = [
        "CREATE INDEX IF NOT EXISTS ix_truck_weigh_orders_truck_id ON truck_weigh_orders (truck_id)",
        "CREATE INDEX IF NOT EXISTS ix_truck_weigh_orders_status ON truck_weigh_orders (status)",
        "CREATE INDEX IF NOT EXISTS ix_truck_weigh_orders_created_at ON truck_weigh_orders (created_at)",
    ]
    column_statements = [
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS silo_no INTEGER",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_delta_kg DOUBLE PRECISION",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_status VARCHAR(20)",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_qty_before DOUBLE PRECISION",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_qty_after DOUBLE PRECISION",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_applied_at TIMESTAMPTZ",
        "ALTER TABLE truck_weigh_orders ADD COLUMN IF NOT EXISTS stock_note VARCHAR(255)",
    ]
    try:
        for stmt in column_statements + index_statements:
            db.session.execute(text(stmt))
        db.session.commit()
    except Exception:
        db.session.rollback()
