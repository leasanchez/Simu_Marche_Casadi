import biorbd

class utils:
    @staticmethod
    def get_q_name(model):
        q_name = []
        for s in range(model.nbSegment()):
            seg_name = model.segment(s).name().to_string()
            for d in range(model.segment(s).nbDof()):
                dof_name = model.segment(s).nameDof(d).to_string()
                q_name.append(seg_name + "_" + dof_name)
        return q_name

    @staticmethod
    def get_q_range(model):
        q_max = []
        q_min = []
        for s in range(model.nbSegment()):
            q_range = model.segment(s).QRanges()
            for r in q_range:
                q_max.append(r.max())
                q_min.append(r.min())
        return q_max, q_min

    @staticmethod
    def get_qdot_range(model):
        qdot_max = []
        qdot_min = []
        for s in range(model.nbSegment()):
            qdot_range = model.segment(s).QDotRanges()
            for r in qdot_range:
                qdot_max.append(r.max())
                qdot_min.append(r.min())
        return qdot_max, qdot_min

    @staticmethod
    def get_contact_name(model):
        contact_name=[]
        C_names = model.contactNames()
        for name in C_names:
            contact_name.append(name.to_string())
        return contact_name

    @staticmethod
    def get_muscle_name(model):
        muscle_name = []
        for m in range(model.nbMuscleTotal()):
            muscle_name.append(model.muscle(m).name().to_string())
        return muscle_name

